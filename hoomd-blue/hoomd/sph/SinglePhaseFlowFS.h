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
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "SmoothingKernel.h"
#include "StateEquations.h"
#include "SPHBaseClass.h"
#include "SinglePhaseFlow.h"
#include "SinglePhaseFlowTV.h"
#include "SolidFluidTypeBit.h"

#include "EvaluationMethodDefinition.h"


/*! \file SinglePhaseFlowFS.h
    \brief Free-surface SPH solver for single-phase flows (thin films, waves, jets).

    Extends SinglePhaseFlowTV with:

    1. Free-surface detection via the Shepard kernel-completeness ratio
           λ_i = V_i·W₀(h) + Σ_{j≠i} V_j·W(r_{ij},h)
       Particles with λ_i < fs_threshold are flagged as free-surface particles.
       Their outward unit normal is n̂_i = −∇λ_i / |∇λ_i|.
       Storage: aux2 (normals, temporarily), aux4.x (lambda), aux4.y (curvature).

    2. Contact-angle enforcement (Huber et al. 2016) at the triple line:
       near solid walls the free-surface normal is blended with the wall normal:
           n̂_corrected = sin(θ_eq)·t̂_w + cos(θ_eq)·n̂_w
       where n̂_w is the wall inward normal and t̂_w the tangential component
       of the gas-side free-surface normal.

    3. Curvature estimation:
           κ_i = (1/V_i) Σ_j V_j (n̂_j − n̂_i)·∇W_{ij}
       Only computed for surface particles; stored in aux4.y.

    4. Free-surface pressure clamping:
       Prevents unphysical tensile pressure at the surface (P ← max(0, P)).

    5. Continuum surface force (CSF) in forcecomputation:
           F_{σ,i} = −σ · κ_i · n̂_i · (m_i/ρ_i)
       The minus sign follows from CSF conventions: for a convex surface
       (κ > 0, n̂ outward), the surface tension pulls inward.

    Aux-array usage summary
    -----------------------
    aux1 : fictitious solid velocity (inherited, Adami 2012)
    aux2 : free-surface outward normal (detect_freesurface → compute_curvature),
           then overwritten by BPC in forcecomputation (as in TV, for KickDriftKickTV)
    aux3 : transport velocity (inherited from TV)
    aux4 : .x = λ (kernel completeness), .y = κ (curvature), .z unused

    References
    ----------
    Marrone et al. (2010), Comput. Fluids — free-surface detection
    Huber et al.  (2016), Int. J. Numer. Meth. Fluids — contact-angle BC
    Colagrossi & Landrini (2003), J. Comput. Phys. — CSF surface tension for SPH
    Adami et al. (2013), J. Comput. Phys. — transport velocity
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __SinglePhaseFlowFS_H__
#define __SinglePhaseFlowFS_H__


namespace hoomd
{
namespace sph
{

/*!  SPH solver with free-surface detection, contact-angle BC, and surface tension.
 *
 *   Inherits the full SinglePhaseFlowTV pipeline (density, pressure, no-slip BC,
 *   TV pair forces, body force, KickDriftKickTV integrator support).
 *   Overrides computeForces() to insert the free-surface pipeline and
 *   overrides forcecomputation() to add the CSF surface tension force.
 */
template<SmoothingKernelType KT_, StateEquationType SET_>
class PYBIND11_EXPORT SinglePhaseFlowFS : public SinglePhaseFlowTV<KT_, SET_>
    {
    public:

        //! Constructor — same signature as SinglePhaseFlowTV.
        SinglePhaseFlowFS(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<SmoothingKernel<KT_>> skernel,
                          std::shared_ptr<StateEquation<SET_>> equationofstate,
                          std::shared_ptr<nsearch::NeighborList> nlist,
                          std::shared_ptr<ParticleGroup> fluidgroup,
                          std::shared_ptr<ParticleGroup> solidgroup,
                          DensityMethod   mdensitymethod  = DENSITYSUMMATION,
                          ViscosityMethod mviscositymethod = HARMONICAVERAGE);

        //! Destructor
        virtual ~SinglePhaseFlowFS();

        /*! Set free-surface parameters.
         *
         * \param sigma          Surface tension coefficient σ [N/m].  Set to 0
         *                       to disable surface tension while keeping
         *                       free-surface detection active.
         * \param fs_threshold   Kernel-completeness threshold λ ∈ (0,1).
         *                       Particles with λ < fs_threshold are treated as
         *                       free-surface particles (typical value: 0.75).
         * \param contact_angle  Equilibrium contact angle θ [rad] measured
         *                       inside the liquid from the solid wall.
         *                       π/2 = neutral wetting (no correction applied),
         *                       0   = complete wetting,
         *                       π   = complete non-wetting.
         */
        void setFSParams(Scalar sigma, Scalar fs_threshold, Scalar contact_angle);

        Scalar getSigma()        { return m_sigma; }
        Scalar getFSThreshold()  { return m_fs_threshold; }
        Scalar getContactAngle() { return m_contact_angle; }

        //! Override: insert free-surface pipeline before forcecomputation
        void computeForces(uint64_t timestep);

    #ifdef ENABLE_MPI
        //! Ghost-communication flags: TV flags + aux4 (lambda / curvature)
        virtual CommFlags getRequestedCommFlags(uint64_t timestep)
            {
            CommFlags flags(0);
            flags[comm_flag::net_force]  = 0;
            flags[comm_flag::position]   = 1;
            flags[comm_flag::velocity]   = 1;
            flags[comm_flag::density]    = 1;
            flags[comm_flag::pressure]   = 1;
            flags[comm_flag::energy]     = 0;
            flags[comm_flag::auxiliary1] = 1;  // fictitious solid velocity
            flags[comm_flag::auxiliary2] = 1;  // BPC / fs normals (shared slot)
            flags[comm_flag::auxiliary3] = 1;  // transport velocity
            flags[comm_flag::auxiliary4] = 1;  // lambda + curvature
            flags[comm_flag::slength]    = 1;
            flags |= ForceCompute::getRequestedCommFlags(timestep);
            return flags;
            }
    #endif

        virtual bool ComputesDPE() { return true; }

    protected:

    #ifdef ENABLE_MPI
        /// MPI communicator (shadowed from TV so FS can access it directly)
        std::shared_ptr<Communicator> m_comm;
    #endif

        // ── Free-surface parameters ───────────────────────────────────────────
        Scalar m_sigma;          //!< Surface tension coefficient σ [N/m]
        Scalar m_fs_threshold;   //!< λ threshold for surface detection
        Scalar m_contact_angle;  //!< Equilibrium contact angle [rad]

        // ── Step 1: detect free surface ───────────────────────────────────────
        /*! Compute kernel-completeness λ_i and outward free-surface normals n̂_i.
         *
         *  Writes:
         *    aux4.x ← λ_i  (Shepard sum including self contribution)
         *    aux2   ← n̂_i  (unit outward normal; {0,0,0} for bulk particles)
         *
         *  When the particle has solid neighbours and |θ − π/2| > 0.01 rad,
         *  the contact-angle correction is applied to n̂_i.
         *
         *  \pre  Density and smoothing lengths are up to date.
         *  \post aux2 and aux4.x are ready for compute_curvature().
         */
        void detect_freesurface(uint64_t timestep);

        // ── Step 2: compute curvature ─────────────────────────────────────────
        /*! Compute mean curvature κ_i = (1/V_i) Σ_j V_j (n̂_j − n̂_i)·∇W_{ij}.
         *
         *  Only executed for surface particles (λ_i < m_fs_threshold).
         *  Writes:
         *    aux4.y ← κ_i
         *
         *  \pre  detect_freesurface() has been called (aux2 holds n̂_i).
         *        In MPI runs, update_ghost_aux24() must be called first so that
         *        ghost-particle normals are available.
         *  \post aux4.y is ready for forcecomputation().
         */
        void compute_curvature(uint64_t timestep);

        // ── Step 3: free-surface pressure clamping ────────────────────────────
        /*! Clamp pressure of free-surface particles to P ≥ 0.
         *
         *  Prevents unphysical tensile pressures that arise from the kernel
         *  truncation near the surface.  Applied only to particles with
         *  λ_i < m_fs_threshold.
         */
        void apply_freesurface_pressure(uint64_t timestep);

        // ── Step 4: force loop with CSF surface tension ───────────────────────
        /*! TV force loop extended with CSF surface tension.
         *
         *  At the start of each particle's outer loop the free-surface normal
         *  is saved from aux2 (written by detect_freesurface), then aux2 is
         *  zeroed for BPC accumulation exactly as in SinglePhaseFlowTV.
         *  After the neighbour loop the CSF surface tension force is added:
         *      F_{σ,i} += −σ · κ_i · n̂_i · (m_i/ρ_i)
         *
         *  \pre  detect_freesurface(), compute_curvature(), and
         *        apply_freesurface_pressure() have been called.
         */
        void forcecomputation(uint64_t timestep);

    #ifdef ENABLE_MPI
        /*! Sync aux2 (fs normals) and aux4 (λ + κ) to ghost particles.
         *
         *  Called between detect_freesurface() and compute_curvature() so that
         *  neighbour normals from other MPI ranks are available in the curvature
         *  loop.
         */
        void update_ghost_aux24(uint64_t timestep);
    #endif

    private:

    };


namespace detail
{
template<SmoothingKernelType KT_, StateEquationType SET_>
void export_SinglePhaseFlowFS(pybind11::module& m, std::string name);

} // end namespace detail
} // end namespace sph
} // end namespace hoomd

#endif // __SinglePhaseFlowFS_H__
