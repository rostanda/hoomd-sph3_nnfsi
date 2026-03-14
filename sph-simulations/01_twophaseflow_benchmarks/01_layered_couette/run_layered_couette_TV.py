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

Two-layer Couette flow — Transport-Velocity (TV) run script.

Uses TwoPhaseFlowTV + KickDriftKickTV instead of the standard VelocityVerletBasic.
The analytical steady-state profile is the same as run_layered_couette.py.

Usage:
    python3 run_layered_couette_TV.py <num_length> <init_gsd_file>
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

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logname   = filename.replace('_init.gsd', '_TV_run.log')
dumpname  = filename.replace('_init.gsd', '_TV_run.gsd')

sim.create_state_from_gsd(filename=filename)

# ─── Physical parameters ─────────────────────────────────────────────────────
lref       = 0.001
U_wall     = 0.01
dx         = lref / num_length
rho01      = 1000.0
rho02      = 1000.0
viscosity1 = 0.004
viscosity2 = 0.001
sigma      = 0.0
backpress  = 0.01
refvel     = U_wall
drho       = 0.01
steps      = 5001

v_i_theory = U_wall * viscosity2 / (viscosity1 + viscosity2)

# ─── Kernel ──────────────────────────────────────────────────────────────────
kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength
kernel_obj = hoomd.sph.kernel.Kernels[kernel]()
kappa      = kernel_obj.Kappa()

nlist = hoomd.nsearch.nlist.Cell(buffer=rcut * 0.05,
                                  rebuild_check_delay=1, kappa=kappa)

# ─── EOS ─────────────────────────────────────────────────────────────────────
eos1 = hoomd.sph.eos.Tait()
eos2 = hoomd.sph.eos.Tait()
eos1.set_params(rho01, backpress)
eos2.set_params(rho02, backpress)

# ─── Filters ─────────────────────────────────────────────────────────────────
filterfluidW = hoomd.filter.Type(['W'])
filterfluidN = hoomd.filter.Type(['N'])
filtersolid  = hoomd.filter.Type(['S'])

# ─── TwoPhaseFlowTV model ────────────────────────────────────────────────────
model = hoomd.sph.sphmodel.TwoPhaseFlowTV(
    kernel=kernel_obj,
    eos1=eos1, eos2=eos2,
    nlist=nlist,
    fluidgroup1_filter=filterfluidW,
    fluidgroup2_filter=filterfluidN,
    solidgroup_filter=filtersolid,
    densitymethod='SUMMATION',
    colorgradientmethod='DENSITYRATIO',
)

model.mu1              = viscosity1
model.mu2              = viscosity2
model.sigma12          = sigma
model.omega            = 90
model.gx               = 0.0
model.damp             = 1000
model.artificialviscosity = True
model.alpha            = 0.2
model.beta             = 0.0
model.densitydiffusion = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c1, cond1, c2, cond2 = model.compute_speedofsound(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length,
    MU1=viscosity1, MU2=viscosity2,
    RHO01=rho01, RHO02=rho02, SIGMA12=sigma)

P_bg_tv1 = backpress * rho01 * c1**2
P_bg_tv2 = backpress * rho02 * c2**2
eos1.set_params(rho01, backpress, tvp=P_bg_tv1)
eos1.set_speedofsound(c1)
eos2.set_params(rho02, backpress, tvp=P_bg_tv2)
eos2.set_speedofsound(c2)

sph_helper.update_min_c0_tpf(device, model, c1, c2,
                              mode='plain', lref=lref, uref=refvel, cfactor=10.0)

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length,
    MU1=viscosity1, MU2=viscosity2,
    RHO01=rho01, RHO02=rho02, SIGMA12=sigma)

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')

# ─── Integrator (KickDriftKickTV) ────────────────────────────────────────────
integrator = hoomd.sph.Integrator(dt=dt)
kdktv_W = hoomd.sph.methods.KickDriftKickTV(filter=filterfluidW,
                                             densitymethod='SUMMATION')
kdktv_N = hoomd.sph.methods.KickDriftKickTV(filter=filterfluidN,
                                             densitymethod='SUMMATION')
integrator.methods.append(kdktv_W)
integrator.methods.append(kdktv_N)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Output ──────────────────────────────────────────────────────────────────
# Remove any stale output GSD left by a previous crashed run.
# ALL ranks attempt the removal so each node's NFS metadata cache is flushed;
# the first rank to run succeeds, the rest get FileNotFoundError (ignored).
try:
    os.remove(dumpname)
except (FileNotFoundError, OSError):
    pass
device.communicator.barrier()

gsd_writer = hoomd.write.GSD(filename=dumpname,
                              trigger=hoomd.trigger.Periodic(50),
                              mode='wb',
                              dynamic=['property', 'momentum'])
sim.operations.writers.append(gsd_writer)

logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps', 'walltime'])
compute_W = hoomd.sph.compute.SinglePhaseFlowBasicProperties(filter=filterfluidW)
compute_N = hoomd.sph.compute.SinglePhaseFlowBasicProperties(filter=filterfluidN)
sim.operations.computes.append(compute_W)
sim.operations.computes.append(compute_N)
logger.add(compute_W, quantities=['e_kin_fluid', 'mean_density'])
logger.add(compute_N, quantities=['e_kin_fluid', 'mean_density'])
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(50), logger=logger,
                          max_header_len=10)
sim.operations.writers.append(table)

log_file = open(logname, mode='w+', newline='\n')
table_file = hoomd.write.Table(output=log_file,
                                trigger=hoomd.trigger.Periodic(50),
                                logger=logger, max_header_len=10)
sim.operations.writers.append(table_file)

# ─── Run ─────────────────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    print(f'Starting TV two-layer Couette run at {dt_string}')
sim.run(steps, write_at_start=True)
gsd_writer.flush()

# ─── Post-processing ─────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    H = lref
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap = traj[-1]
        pos  = snap.particles.position
        tid  = snap.particles.typeid
        vel  = snap.particles.velocity

    fluid = (tid == 0) | (tid == 1)
    y_f   = pos[fluid, 1]
    vx_f  = vel[fluid, 0]

    def v_analytical(y):
        v = np.zeros_like(y)
        lower = y < 0.0
        upper = y >= 0.0
        v[lower] = v_i_theory * (y[lower] + H / 2) / (H / 2)
        v[upper] = v_i_theory + (U_wall - v_i_theory) * y[upper] / (H / 2)
        return v

    in_fluid = np.abs(y_f) < H / 2
    y_ev  = y_f[in_fluid]
    vx_ev = vx_f[in_fluid]
    va_ev = v_analytical(y_ev)

    L2_err = np.sqrt(np.mean((vx_ev - va_ev)**2)) / U_wall * 100.0

    print(f'\n── Couette TV profile check (last frame, step {snap.configuration.step}) ──')
    print(f'  Analytical v_i   = {v_i_theory:.5f} m/s')
    print(f'  L₂ error / U_wall = {L2_err:.2f} %')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_tpftv(dumpname)
