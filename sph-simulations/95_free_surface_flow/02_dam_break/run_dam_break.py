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

2-D dam-break — SinglePhaseFlowFS benchmark.

BENCHMARK DESCRIPTION
---------------------
A rectangular water column (width a = lref, height H₀ = 2a) collapses
under gravity (gy = -9.81 m/s²) into an empty channel of total length
L = 4a.  Solid walls bound the floor and channel ends; the top and the
right side of the initial column are free surfaces.

Analytical reference — shallow-water front position (Martin & Moyce 1952):
    X*(T*) = x_front / a  ≈  1 + 2√2 · T*
where the dimensionless time is
    T* = t · √(g / a)
and √2 factor comes from wave speed c₀ = √(g H₀) = √(g · 2a) = √2 · √(ga).

Post-processing reads every saved GSD frame, computes the maximum x
position of fluid particles (= front position x_front), and prints
X* vs T* alongside the shallow-water prediction.

Usage:
    python3 run_dam_break.py <num_length> <init_gsd_file> [steps]
    e.g.  python3 run_dam_break.py 20 dam_break_*_init.gsd 10001
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
lref      = 0.01            # dam-column width a                    [m]
H0        = 2.0 * lref     # initial column height                  [m]
dx        = lref / num_length
rho0      = 1000.0          # rest density                          [kg/m³]
viscosity = 1e-3            # dynamic viscosity (water)             [Pa·s]
gy        = -9.81           # gravitational acceleration            [m/s²]
drho      = 0.01
backpress = 0.01
nu        = viscosity / rho0
c0_wave   = np.sqrt(abs(gy) * H0)  # shallow-water wave speed      [m/s]
refvel    = c0_wave

# ─── Free-surface parameters ──────────────────────────────────────────────────
sigma         = 0.0         # no surface tension for dam-break
fs_threshold  = 0.75
contact_angle = np.pi / 2

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
model.damp                = 0              # no artificial damping for dynamics
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
    print(f'Shallow-water wave speed c₀ = {c0_wave:.4f} m/s')

sph_helper.update_min_c0(device, model, c,
                          mode='uref', lref=lref, uref=refvel, cfactor=10.0)

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')
    T_star_end = steps * dt * np.sqrt(abs(gy) / lref)
    print(f'Simulated T* range: [0, {T_star_end:.2f}]  (shallow water theory valid for T* < ~3)')

# ─── Integrator (KickDriftKickTV — required for SinglePhaseFlowFS) ────────────
integrator = hoomd.sph.Integrator(dt=dt)
kdktv = hoomd.sph.methods.KickDriftKickTV(filter=filterfluid,
                                           densitymethod='SUMMATION')
integrator.methods.append(kdktv)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Output ──────────────────────────────────────────────────────────────────
# Write more frequently so the front-position time series is well resolved
write_period = max(100, steps // 200)
gsd_writer = hoomd.write.GSD(filename=dumpname,
                              trigger=hoomd.trigger.Periodic(write_period),
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
    print(f'Starting 2-D dam-break run at {dt_string}')
sim.run(steps, write_at_start=True)

# ─── Post-processing: front position X*(T*) vs shallow-water theory ──────────
if device.communicator.rank == 0:
    g     = abs(gy)
    a     = lref
    x_front_arr = []
    t_arr       = []

    with gsd.hoomd.open(dumpname, 'r') as traj:
        # Get initial x position of the right edge of the fluid column
        snap0    = traj[0]
        fluid0   = (snap0.particles.typeid == 0)
        x_front0 = float(np.max(snap0.particles.position[fluid0, 0]))

        for snap in traj:
            fluid    = (snap.particles.typeid == 0)
            x_front  = float(np.max(snap.particles.position[fluid, 0]))
            t_phys   = snap.configuration.step * dt
            x_front_arr.append(x_front)
            t_arr.append(t_phys)

    x_front_arr = np.array(x_front_arr)
    t_arr       = np.array(t_arr)

    # Normalise: X* = x_front / a,  T* = t * sqrt(g / a)
    X_star_sph = (x_front_arr - x_front0 + a) / a   # shift so X*(0) = 1
    T_star     = t_arr * np.sqrt(g / a)
    X_star_an  = 1.0 + 2.0 * np.sqrt(2.0) * T_star  # shallow-water theory

    # L2 error over all frames where T* <= 3 (theory is valid early-time)
    early = T_star <= 3.0
    if np.any(early):
        L2_front = np.sqrt(np.mean((X_star_sph[early] - X_star_an[early])**2))
    else:
        L2_front = float('nan')

    print(f'\n── Dam-break front-position check (last frame, step {int(t_arr[-1]/dt)}) ──')
    print(f'  a = {a*1e3:.1f} mm,  H₀ = {H0*1e3:.1f} mm')
    print(f'  T* at end     = {T_star[-1]:.2f}')
    print(f'  X* SPH (end)  = {X_star_sph[-1]:.4f}')
    print(f'  X* theory     = {X_star_an[-1]:.4f}')
    print(f'  L₂ error X* (T* ≤ 3) = {L2_front:.4f}  (lower is better)')
    print()
    print('  T*     X*_SPH   X*_theory')
    # Print ~10 representative frames
    idx_print = np.unique(np.linspace(0, len(T_star) - 1, 12, dtype=int))
    for i in idx_print:
        print(f'  {T_star[i]:5.2f}  {X_star_sph[i]:8.4f}  {X_star_an[i]:8.4f}')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_fs(dumpname)
