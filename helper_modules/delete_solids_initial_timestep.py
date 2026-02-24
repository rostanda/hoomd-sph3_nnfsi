#!/usr/bin/env python3

"""
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

"""

# --- HEADER -------------------------------------------------
import hoomd
from hoomd import *
from hoomd import sph
import numpy as np
import itertools
import gsd.hoomd
# ------------------------------------------------------------


def delete_solids(sim, device, kernel, dt, mu, DX, rho0):
    """Remove redundant solid particles from a pre-built simulation.

    When a porous medium geometry is created from a raw (voxel) file, solid
    particles that are entirely surrounded by other solids (i.e. not in
    contact with the fluid) are tagged with a sentinel mass value of -999.
    This function runs a single SPH step to populate the SPH bookkeeping
    arrays, then removes all solid particles carrying that sentinel value.
    This reduces memory and computational cost for subsequent production runs.

    A tiny body force (gx = 1e-7) is applied rather than zero because some
    internal SPH routines skip computation for particles with no applied
    force; the value is negligible compared to any physical body force.

    Parameters
    ----------
    sim : hoomd.Simulation
        Active HOOMD simulation object containing the full (un-pruned) system.
    device : hoomd.device.Device
        HOOMD device; used for the MPI barrier and rank-0 output.
    kernel : str
        Kernel name, one of ``'WendlandC2'``, ``'WendlandC4'``,
        ``'WendlandC6'``, ``'Quintic'``, ``'CubicSpline'``.
    dt : float
        Time-step size used for the single preliminary run step [s].
    mu : float
        Dynamic viscosity of the fluid [Pa·s].
    DX : float
        Initial particle spacing [m].
    rho0 : float
        Reference fluid density [kg/m³].

    Returns
    -------
    sim : hoomd.Simulation
        The simulation object after particle removal.
    deleted : int
        Number of redundant solid particles removed on this MPI rank.
    """

    # Some additional parameters
    densitymethod = 'CONTINUITY'

    # Kernel
    KERNEL  = str(kernel)
    H       = hoomd.sph.kernel.OptimalH[KERNEL]*DX       # m
    RCUT    = hoomd.sph.kernel.Kappa[KERNEL]*H           # m
    Kernel = hoomd.sph.kernel.Kernels[KERNEL]()
    Kappa  = Kernel.Kappa()

    # Neighbor list
    NList = hoomd.nsearch.nlist.Cell(buffer=RCUT*0.05, rebuild_check_delay=1, kappa=Kappa)

    # Setup all necessary simulation inputs
    EOS = hoomd.sph.eos.Linear()
    EOS.set_params(rho0, 0.01)

    # Define groups/filters
    filterFLUID  = hoomd.filter.Type(['F'])
    filterSOLID  = hoomd.filter.Type(['S'])

    # Set up SPH solver
    model = hoomd.sph.sphmodel.SinglePhaseFlow(kernel=Kernel,
                                               eos=EOS,
                                               nlist=NList,
                                               fluidgroup_filter=filterFLUID,
                                               solidgroup_filter=filterSOLID)

    model.mu = mu
    model.densitymethod = densitymethod
    # Tiny body force avoids zero-force code paths in the SPH kernel; value
    # is negligible and has no physical effect on the pruning step.
    model.gx = 0.0000001
    model.damp = 1000
    model.artificialviscosity = False
    model.densitydiffusion = False
    model.shepardrenormanlization = False

    # Define integrator and run one step to populate internal arrays.
    integrator = hoomd.sph.Integrator(dt=dt)
    VelocityVerlet = hoomd.sph.methods.VelocityVerletBasic(filter=filterFLUID, densitymethod=densitymethod)

    sim.operations.integrator = integrator
    integrator.methods.append(VelocityVerlet)
    integrator.forces.append(model)

    sim.run(1, write_at_start=False)

    # Identify redundant solid particles by the sentinel mass value -999.
    # This sentinel is written by the geometry-creation script for solid
    # particles that have no fluid neighbours and can therefore be discarded.
    tags    = []
    deleted = 0

    with sim.state.cpu_local_snapshot as data:
        for i in range(len(data.particles.position)):
            if data.particles.typeid[i] == 1 and data.particles.mass[i] == -999:
                tags.append(data.particles.tag[i])
                deleted += 1

    for t in tags:
        sim.state.removeParticle(t)

    device.communicator.barrier_all()

    print(f'Rank {device.communicator.rank}: {deleted} unnecessary solid particles deleted.')

    return sim, deleted

