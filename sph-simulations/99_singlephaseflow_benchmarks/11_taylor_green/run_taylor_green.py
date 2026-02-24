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

2-D Taylor–Green vortex decay — WCSPH benchmark.

BENCHMARK DESCRIPTION
---------------------
A fully periodic 2-D square domain with the Taylor–Green initial condition:
    v_x = -U0 * cos(kx) * sin(ky)
    v_y =  U0 * sin(kx) * cos(ky)
where k = 2π/L and L = lref.

The analytical solution is an exponentially decaying vortex:
    v_x(x, y, t) = -U0 * cos(kx) * sin(ky) * exp(-2 ν k² t)
    v_y(x, y, t) =  U0 * sin(kx) * cos(ky) * exp(-2 ν k² t)

This benchmark tests:
  - Conservation of kinetic energy in a periodic domain
  - Correct viscous dissipation rate
  - Absence of numerical diffusion artefacts

Post-processing computes:
  - Total kinetic energy vs analytical E(t) = E0 * exp(-4 ν k² t)
  - L2 velocity error relative to the analytical field at the final snapshot

Usage:
    python3 run_taylor_green.py <num_length> <init_gsd_file> [steps] [U0]
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
U0         = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logname   = filename.replace('_init.gsd', '_runWC.log')
dumpname  = filename.replace('_init.gsd', '_runWC.gsd')

sim.create_state_from_gsd(filename=filename)

# ─── Physical parameters ─────────────────────────────────────────────────────
lref       = 1.0            # domain side length                    [m]
dx         = lref / num_length
rho0       = 1.0            # rest density                          [kg/m³]
viscosity  = 0.01           # dynamic viscosity                     [Pa·s]
drho       = 0.05
backpress  = 0.01
nu         = viscosity / rho0
k          = 2.0 * np.pi / lref  # wavenumber
refvel     = U0             # reference velocity                    [m/s]
t_decay    = 1.0 / (2.0 * nu * k**2)  # e-folding time [s]

if device.communicator.rank == 0:
    print(f'Taylor-Green vortex: L={lref}, U0={U0}, nu={nu:.4e}')
    print(f'  Viscous decay time t* = {t_decay:.4f} s')

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

# ─── Filters (no solid particles in this benchmark) ──────────────────────────
filterfluid = hoomd.filter.Type(['F'])
filtersolid = hoomd.filter.Type(['S'])

# ─── SPH model ───────────────────────────────────────────────────────────────
model = hoomd.sph.sphmodel.SinglePhaseFlow(
    kernel=kernel_obj, eos=eos, nlist=nlist,
    fluidgroup_filter=filterfluid, solidgroup_filter=filtersolid,
    densitymethod='SUMMATION')

model.mu               = viscosity
model.gx               = 0.0
model.damp             = 0          # no damping — free decay test
model.artificialviscosity = False   # no AV — pure physical viscosity
model.densitydiffusion = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c, cond = model.compute_speedofsound(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

if device.communicator.rank == 0:
    print(f'Speed of sound: {c:.4f} m/s  ({cond})')

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
    print(f'Starting Taylor-Green vortex run at {dt_string}')
sim.run(steps, write_at_start=True)

# ─── Post-processing: kinetic energy and L2 velocity error ───────────────────
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap_0 = traj[0]
        snap_f = traj[-1]

    t_final = snap_f.configuration.step * dt

    # Analytical decay factor at final time
    decay = np.exp(-2.0 * nu * k**2 * t_final)

    pos_f  = snap_f.particles.position
    vel_f  = snap_f.particles.velocity
    x_f    = pos_f[:, 0]
    y_f    = pos_f[:, 1]
    vx_f   = vel_f[:, 0]
    vy_f   = vel_f[:, 1]

    # Analytical velocity at particle positions
    vx_an  = -U0 * decay * np.cos(k * x_f) * np.sin(k * y_f)
    vy_an  =  U0 * decay * np.sin(k * x_f) * np.cos(k * y_f)

    E_sph = 0.5 * float(np.mean(vx_f**2 + vy_f**2))
    E_an  = 0.5 * U0**2 * decay**2  # E0 * exp(-4 nu k^2 t)

    L2_v = np.sqrt(np.mean((vx_f - vx_an)**2 + (vy_f - vy_an)**2)) / U0 * 100.0

    print(f'\n── Taylor–Green vortex check (final step {snap_f.configuration.step}, t={t_final:.4f} s) ──')
    print(f'  Analytical decay factor exp(-2νk²t) = {decay:.6f}')
    print(f'  Analytical E(t)  = {E_an:.6f}  (= E0 * {decay**2:.6f})')
    print(f'  SPH        E(t)  = {E_sph:.6f}')
    print(f'  L₂ velocity error / U0 = {L2_v:.2f} %')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_spf(dumpname)
