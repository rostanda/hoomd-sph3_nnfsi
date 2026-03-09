/* ---------------------------------------------------------
Copyright (c) 2025-2026 David Krach, Daniel Rostan.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

maintainer: dkrach, david.krach@mib.uni-stuttgart.de
----------------------------------------------------------*/

#include "hoomd/Compute.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/Integrator.h"
#include "hoomd/nsearch/NeighborList.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

#include <memory>
#include <vector>
#include <stdexcept>
#include <utility>
#include <set>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "SmoothingKernel.h"
#include "StateEquations.h"
#include "SPHBaseClass.h"
#include "TwoPhaseFlow.h"
#include "SolidFluidTypeBit.h"

#include "EvaluationMethodDefinition.h"

/*! \file TwoPhaseFlowTV.h
    \brief Quasi-incompressible two-phase Navier–Stokes SPH solver with
           transport-velocity (TV) formulation (Adami et al. 2013).

    Extends TwoPhaseFlow with three additional terms in the momentum equation:

    1. Artificial stress  (Adami 2013):
         \f$A_i = \rho_i v_i \otimes (tv_i - v_i)\f$
         \f$\sum_j (V_i^2+V_j^2) (A_i+A_j) \cdot \nabla W_{ij}\f$  added to the force.
         Suppresses the tensile instability (particle clustering) by penalising
         deviations of the transport velocity from the physical velocity.

    2. Background-pressure contribution (BPC):
         \f$\sum_j (V_i^2+V_j^2) P_\mathrm{bg} / m_i \cdot \nabla W_{ij}\f$  accumulated in aux2.
         Read by KickDriftKickTV on the next half-step to advect particles with
         the transport velocity rather than the physical velocity.
         P_bg = StateEquation::getTransportVelocityPressure() (set per EOS).

    Inherited physics (all toggleable independently):
    3. Consistent interface pressure (Hu & Adams 2009) — see TwoPhaseFlow.
    4. Riemann-based dissipation (Zhang, Hu & Adams 2017) — see TwoPhaseFlow.
    5. Corrected density diffusion for two-phase flows — see TwoPhaseFlow.

    Array layout during forcecomputation():
      aux1 : fictitious solid velocities  (read; set by compute_noslip)
      aux2 : BPC accumulator              (written here; read by KickDriftKickTV)
      aux3 : transport velocity           (read; restored from m_tv_buf before call)
      aux4 : surface force density        (read; set by compute_surfaceforce)

    Array conflict and resolution:
      TwoPhaseFlow uses aux2 for solid-fluid normals and aux3 for fluid-fluid normals
      inside compute_colorgradients() / compute_surfaceforce().  TwoPhaseFlowTV needs
      aux3 to carry the transport velocity across the step.  The conflict is resolved
      in computeForces() by saving aux3 into m_tv_buf before compute_colorgradients(),
      then restoring it (and zeroing aux2) immediately before forcecomputation().

    References:
      Adami, Hu & Adams (2013) J. Comput. Phys. 241, 292–307 — transport velocity.
      Hu & Adams (2009) J. Comput. Phys. 228(20), 7518–7530 — CIP.
      Zhang, Hu & Adams (2017) J. Comput. Phys. 340, 439–455 — Riemann dissipation.
      Molteni & Colagrossi (2009) Comput. Phys. Commun. 180, 861–872 — density diffusion.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __TwoPhaseFlowTV_H__
#define __TwoPhaseFlowTV_H__

namespace hoomd
{
namespace sph
{

//! Two-phase SPH force compute with transport-velocity formulation.
/*!
 *  Subclass of TwoPhaseFlow that adds the Adami 2013 transport-velocity (TV)
 *  terms to the momentum equation.  All features of TwoPhaseFlow (CIP, Riemann
 *  dissipation, corrected density diffusion) are inherited and remain toggleable.
 *
 *  Extra momentum terms added by this class (fluid–fluid pairs only):
 *
 *    Artificial stress (Adami 2013, Eq. 11):
 *      \f$F_i^{AS} \mathrel{+}= (1/2) \sum_j (V_i^2+V_j^2) (A_i + A_j) \cdot \nabla W_{ij}\f$
 *      \f$A_k = \rho_k v_k \otimes (tv_k - v_k)\f$   [rank-2 tensor, built from outer product]
 *
 *    Background pressure contribution (BPC, stored in aux2):
 *      \f$\mathrm{bpc}_i \mathrel{+}= -\sum_j (V_i^2+V_j^2) P_\mathrm{bg} / m_i \cdot (\partial W/\partial r)/r \cdot r_{ij}\f$
 *      where P_bg = EOS::getTransportVelocityPressure() (a positive constant
 *      large enough to keep all particle pressures positive).
 *      KickDriftKickTV reads bpc_i on the next integrator half-step to
 *      advance particle positions with the transport velocity tv rather than v.
 *
 *  See file-level documentation in TwoPhaseFlowTV.h for the array layout and
 *  the aux3 save/restore mechanism.
 */
template<SmoothingKernelType KT_, StateEquationType SET1_, StateEquationType SET2_>
class PYBIND11_EXPORT TwoPhaseFlowTV : public TwoPhaseFlow<KT_, SET1_, SET2_>
    {
    public:
        //! Constructor
        TwoPhaseFlowTV(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<SmoothingKernel<KT_> > skernel,
                       std::shared_ptr<StateEquation<SET1_> > equationofstate1,
                       std::shared_ptr<StateEquation<SET2_> > equationofstate2,
                       std::shared_ptr<nsearch::NeighborList> nlist,
                       std::shared_ptr<ParticleGroup> fluidgroup1,
                       std::shared_ptr<ParticleGroup> fluidgroup2,
                       std::shared_ptr<ParticleGroup> solidgroup,
                       DensityMethod   mdensitymethod=DENSITYSUMMATION,
                       ViscosityMethod mviscositymethod=HARMONICAVERAGE,
                       ColorGradientMethod mcolorgradientmethod=DENSITYRATIO);

        //! Destructor
        virtual ~TwoPhaseFlowTV();

        //! Computes forces (full override with TV save/restore)
        virtual void computeForces(uint64_t timestep);

    #ifdef ENABLE_MPI
        //! Get requested ghost communication flags
        virtual CommFlags getRequestedCommFlags(uint64_t timestep)
            {
            CommFlags flags = TwoPhaseFlow<KT_, SET1_, SET2_>::getRequestedCommFlags(timestep);
            flags[comm_flag::auxiliary2] = 2; // background pressure contribution
            flags[comm_flag::auxiliary3] = 1; // transport velocity
            return flags;
            }
    #endif

        //! Returns true because we compute dpe array content
        virtual bool ComputesDPE()
            {
            return true;
            }

    protected:
        //! Scratch buffer holding aux3 (transport velocity) during the colour-gradient
        //! phase.  Sized to getN() + getNGhosts() in computeForces(); valid only
        //! between the save (before compute_colorgradients) and restore (before
        //! forcecomputation) calls.
        std::vector<Scalar3> m_tv_buf;

        /*! Momentum-force kernel — modified copy of TwoPhaseFlow::forcecomputation()
         *  with added artificial-stress and background-pressure-contribution terms.
         *
         *  Called by computeForces() after the following pre-conditions are met:
         *    aux1 = fictitious solid velocities (compute_noslip)
         *    aux2 = zeroed (BPC accumulator, ready to receive contributions)
         *    aux3 = transport velocity (restored from m_tv_buf)
         *    aux4 = surface force density (compute_surfaceforce)
         *
         *  On return:
         *    h_force += pressure + viscous + artificial-stress + surface forces
         *    aux2    = background pressure contribution (read by KickDriftKickTV)
         */
        void forcecomputation(uint64_t timestep);

    private:

    };


namespace detail
{
template<SmoothingKernelType KT_, StateEquationType SET1_, StateEquationType SET2_>
void export_TwoPhaseFlowTV(pybind11::module& m, std::string name);

} // end namespace detail
} // end namespace sph
} // end namespace hoomd

#endif // __TwoPhaseFlowTV_H__
