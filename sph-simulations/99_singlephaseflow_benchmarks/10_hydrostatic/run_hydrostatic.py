#!/usr/bin/env python3
"""
Copyright (c) 2025-2026 David Krach, Daniel Rostan.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

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
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

maintainer: dkrach, david.krach@mib.uni-stuttgart.de

Hydrostatic pressure column — WCSPH benchmark.

BENCHMARK DESCRIPTION
---------------------
A fluid column of height lref is in hydrostatic equilibrium under body
force gy (gravity in negative y direction) between two solid plates.

Analytical solution (hydrostatic pressure):
    p(y) = p0 + rho0 * |gy| * (lref/2 - y)

This tests:
  - Correct EOS pressure initialisation and steady-state density distribution
  - Absence of spurious velocity (should remain < 1e-4 * sqrt(|gy|*lref))
  - Solid-wall boundary conditions at the bottom plate

Post-processing computes the L2 error of the density profile vs the analytical
hydrostatically compressed profile:
    rho(y) = rho0 * (1 + rho0*|gy|*(lref/2-y) / (c0^2 * backpress))^(1/gamma)
for Tait EOS, or approximately rho ≈ rho0 for weakly compressible SPH.

Usage:
    python3 run_hydrostatic.py <num_length> <init_gsd_file> [steps]
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, os.path.join(_ROOT, 'helper_modules'))
sys.path.insert(0, os.path.join(_ROOT, 'helper_modules', 'gsd2vtu'))

import hoomd
from hoomd import sph
import numpy as np
from datetime import datetime
import gsd.hoomd
import sph_helper

try:
    import export_gsd2vtu
    HAS_VTU = True
except ImportError:
    HAS_VTU = False

# ─── Device & simulation ─────────────────────────────────────────────────────
device = hoomd.device.CPU(notice_level=2)
sim    = hoomd.Simulation(device=device)

num_length = int(sys.argv[1])
filename   = str(sys.argv[2])
steps      = int(sys.argv[3]) if len(sys.argv) > 3 else 10001

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logname   = filename.replace('_init.gsd', '_runWC.log')
dumpname  = filename.replace('_init.gsd', '_runWC.gsd')

sim.create_state_from_gsd(filename=filename)

# ─── Physical parameters ─────────────────────────────────────────────────────
lref       = 0.001          # column height                         [m]
dx         = lref / num_length
rho0       = 1000.0         # rest density                          [kg/m³]
viscosity  = 0.01           # dynamic viscosity                     [Pa·s]
gy         = -9.81          # gravitational acceleration (negative y) [m/s²]
drho       = 0.01
backpress  = 0.01
nu         = viscosity / rho0
# Reference velocity: buoyancy-driven scale sqrt(|gy|*lref)
refvel     = np.sqrt(abs(gy) * lref)   # ~ 0.099 m/s

# ─── Kernel ──────────────────────────────────────────────────────────────────
kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength
kernel_obj = hoomd.sph.kernel.Kernels[kernel]()
kappa      = kernel_obj.Kappa()

nlist = hoomd.nsearch.nlist.Cell(buffer=rcut * 0.05,
                                  rebuild_check_delay=1, kappa=kappa)

# ─── EOS ─────────────────────────────────────────────────────────────────────
eos = hoomd.sph.eos.Tait()
eos.set_params(rho0, backpress)

# ─── Filters ─────────────────────────────────────────────────────────────────
filterfluid = hoomd.filter.Type(['F'])
filtersolid = hoomd.filter.Type(['S'])

# ─── SPH model ───────────────────────────────────────────────────────────────
model = hoomd.sph.sphmodel.SinglePhaseFlow(
    kernel=kernel_obj, eos=eos, nlist=nlist,
    fluidgroup_filter=filterfluid, solidgroup_filter=filtersolid,
    densitymethod='SUMMATION')

model.mu               = viscosity
model.gx               = 0.0
model.gy               = gy
model.damp             = 5000
model.artificialviscosity = True
model.alpha            = 0.1
model.beta             = 0.0
model.densitydiffusion = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c, cond = model.compute_speedofsound(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

if device.communicator.rank == 0:
    print(f'Speed of sound: {c:.4f} m/s  ({cond})')
    print(f'Analytical pressure head = {rho0 * abs(gy) * lref:.4f} Pa')

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')

# ─── Integrator ──────────────────────────────────────────────────────────────
integrator = hoomd.sph.Integrator(dt=dt)
vvb = hoomd.sph.methods.VelocityVerletBasic(filter=filterfluid,
                                              densitymethod='SUMMATION')
integrator.methods.append(vvb)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Output ──────────────────────────────────────────────────────────────────
gsd_writer = hoomd.write.GSD(filename=dumpname,
                              trigger=hoomd.trigger.Periodic(1000),
                              mode='wb',
                              dynamic=['property', 'momentum'])
sim.operations.writers.append(gsd_writer)

logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps', 'walltime'])
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(500), logger=logger,
                          max_header_len=10)
sim.operations.writers.append(table)

log_file = open(logname, mode='w+', newline='\n')
table_file = hoomd.write.Table(output=log_file,
                                trigger=hoomd.trigger.Periodic(500),
                                logger=logger, max_header_len=10)
sim.operations.writers.append(table_file)

# ─── Run ─────────────────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    print(f'Starting hydrostatic column run at {dt_string}')
sim.run(steps, write_at_start=True)

# ─── Post-processing: pressure profile vs analytical ─────────────────────────
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap = traj[-1]
        pos  = snap.particles.position
        tid  = snap.particles.typeid
        vel  = snap.particles.velocity
        den  = snap.particles.density

    fluid   = (tid == 0)
    y_f     = pos[fluid, 1]
    rho_f   = den[fluid]
    vx_f    = vel[fluid, 0]
    vy_f    = vel[fluid, 1]

    # Analytical hydrostatic density for weakly compressible SPH (Tait EOS):
    #   p = c0^2 * backpress * rho0/gamma * ((rho/rho0)^gamma - 1)
    # For the pressure check, use linear approximation:
    #   rho(y) ≈ rho0 * (1 + rho0*|gy|*(H/2-y) / p_ref)
    # where p_ref = c0^2 * rho0 (bulk modulus approximation)
    p_ref     = c**2 * rho0
    rho_an    = rho0 * (1.0 + rho0 * abs(gy) * (0.5 * lref - y_f) / p_ref)
    L2_rho    = np.sqrt(np.mean((rho_f - rho_an)**2)) / rho0 * 100.0

    v_spurious = float(np.sqrt(np.mean(vx_f**2 + vy_f**2)))

    print(f'\n── Hydrostatic column check (last frame, step {snap.configuration.step}) ──')
    print(f'  Analytical Δp (top→bottom) = {rho0 * abs(gy) * lref:.4f} Pa')
    print(f'  L₂ density error vs hydrostatic = {L2_rho:.3f} %')
    print(f'  RMS spurious velocity            = {v_spurious:.3e} m/s')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_spf(dumpname)
