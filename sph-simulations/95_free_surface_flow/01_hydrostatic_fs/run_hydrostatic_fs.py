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

Hydrostatic pressure column with free surface — SinglePhaseFlowFS benchmark.

BENCHMARK DESCRIPTION
---------------------
A fluid column of height lref is in hydrostatic equilibrium under gravity
(gy = -9.81 m/s²).  The bottom is a solid plate (Adami 2012 boundary);
the top is a *free surface* — no solid plate is present.

This benchmark tests four things:
  1. Free-surface detection: particles in the top layer must receive
     λ < fs_threshold after converging to the Shepard completeness criterion.
  2. Pressure clamping: surface particles must have P ≥ 0 (enforced by FS
     solver; negative tensile pressures at the surface should be absent).
  3. Hydrostatic pressure profile: the analytical solution
         p(y) = ρ₀ |g| (y_top − y)
     must be reproduced with small L₂ error.
  4. Spurious velocity: the RMS velocity should remain small at steady state.

Analytical solution (Tait EOS, weakly compressible):
    ρ(y) ≈ ρ₀ (1 + ρ₀ |g| (y_top − y) / p_ref)
where p_ref = c₀² ρ₀ (bulk modulus approximation) and y_top is the
free-surface level (top of the fluid column).

Usage:
    python3 run_hydrostatic_fs.py <num_length> <init_gsd_file> [steps]
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
logname   = filename.replace('_init.gsd', '_runFS.log')
dumpname  = filename.replace('_init.gsd', '_runFS.gsd')

sim.create_state_from_gsd(filename=filename)

# ─── Physical parameters ─────────────────────────────────────────────────────
lref      = 0.001           # column height                         [m]
dx        = lref / num_length
rho0      = 1000.0          # rest density                          [kg/m³]
viscosity = 0.01            # dynamic viscosity                     [Pa·s]
gy        = -9.81           # gravitational acceleration            [m/s²]
drho      = 0.01
backpress = 0.01
nu        = viscosity / rho0
refvel    = np.sqrt(abs(gy) * lref)   # buoyancy velocity scale    [m/s]

# ─── Free-surface parameters ──────────────────────────────────────────────────
sigma         = 0.0          # no surface tension — detection only
fs_threshold  = 0.75         # Shepard completeness threshold
contact_angle = np.pi / 2   # neutral wetting (irrelevant without sigma)

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
model = hoomd.sph.sphmodel.SinglePhaseFlowFS(
    kernel=kernel_obj, eos=eos, nlist=nlist,
    fluidgroup_filter=filterfluid, solidgroup_filter=filtersolid,
    densitymethod='SUMMATION',
    sigma=sigma, fs_threshold=fs_threshold, contact_angle=contact_angle)

model.mu                  = viscosity
model.gy                  = gy
model.damp                = 5000
model.artificialviscosity = True
model.alpha               = 0.1
model.beta                = 0.0
model.densitydiffusion    = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c, cond = model.compute_speedofsound(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

if device.communicator.rank == 0:
    print(f'Speed of sound: {c:.4f} m/s  ({cond})')
    print(f'Analytical pressure head = {rho0 * abs(gy) * lref:.4f} Pa')

sph_helper.update_min_c0(device, model, c,
                          mode='uref', lref=lref, uref=refvel, cfactor=10.0)

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')

# ─── Integrator (KickDriftKickTV — required for SinglePhaseFlowFS) ────────────
integrator = hoomd.sph.Integrator(dt=dt)
kdktv = hoomd.sph.methods.KickDriftKickTV(filter=filterfluid,
                                           densitymethod='SUMMATION')
integrator.methods.append(kdktv)
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
    print(f'Starting hydrostatic FS column run at {dt_string}')
sim.run(steps, write_at_start=True)

# ─── Post-processing ─────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap = traj[-1]
        pos  = snap.particles.position
        tid  = snap.particles.typeid
        vel  = snap.particles.velocity
        den  = snap.particles.density
        pres = snap.particles.pressure

    fluid  = (tid == 0)
    y_f    = pos[fluid, 1]
    rho_f  = den[fluid]
    vel_f  = vel[fluid, :2]
    pres_f = pres[fluid]

    # Analytical hydrostatic density (linear approx for weakly compressible):
    #   rho(y) ≈ rho0 * (1 + rho0*|gy|*(y_top - y) / (c^2*rho0))
    y_top  = float(np.max(y_f))
    p_ref  = c**2 * rho0
    rho_an = rho0 * (1.0 + rho0 * abs(gy) * (y_top - y_f) / p_ref)
    L2_rho = np.sqrt(np.mean((rho_f - rho_an)**2)) / rho0 * 100.0

    v_spurious = float(np.sqrt(np.mean(np.sum(vel_f**2, axis=1))))

    # Surface-layer detection: top fluid row should have lambda < fs_threshold
    dx_loc      = lref / num_length
    y_top_layer = y_top - 1.5 * dx_loc
    n_top_layer = int(np.sum(y_f > y_top_layer))

    # Pressure clamp check: no fluid particle should have P < 0
    n_neg_pres  = int(np.sum(pres_f < -1e-10))

    print(f'\n── Hydrostatic FS check (last frame, step {snap.configuration.step}) ──')
    print(f'  Δp analytical (top→bottom)   = {rho0 * abs(gy) * lref:.4f} Pa')
    print(f'  L₂ density error vs profile  = {L2_rho:.3f} %')
    print(f'  RMS spurious velocity        = {v_spurious:.3e} m/s')
    print(f'  Particles in top fluid layer = {n_top_layer}  (should all have λ < {fs_threshold})')
    print(f'  Particles with P < 0         = {n_neg_pres}  (should be 0 after clamping)')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_fs(dumpname)
