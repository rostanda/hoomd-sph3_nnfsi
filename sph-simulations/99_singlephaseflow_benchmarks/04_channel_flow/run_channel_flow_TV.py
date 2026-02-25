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

Square-duct channel flow — Transport-Velocity (TV) run script.

Uses SinglePhaseFlowTV + KickDriftKickTV.  The analytical centreline velocity
for a square duct is the same as run_channel_flow.py:
    $v_\mathrm{cl} \approx 0.2947 \times f_x \times a^2 / \mu$

Usage:
    python3 run_channel_flow_TV.py <num_length> <init_gsd_file> [steps]
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
logname   = filename.replace('_init.gsd', '_runTV.log')
dumpname  = filename.replace('_init.gsd', '_runTV.gsd')

sim.create_state_from_gsd(filename=filename, domain_decomposition=(None, None, 1))

# ─── Physical parameters ─────────────────────────────────────────────────────
lref       = 0.001
a          = 0.5 * lref
dx         = lref / num_length
rho0       = 1000.0
viscosity  = 0.01
fx         = 0.1
drho       = 0.01
backpress  = 0.01
nu         = viscosity / rho0
v_cl_an    = 0.2947 * fx * a**2 / viscosity
refvel     = v_cl_an

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
model = hoomd.sph.sphmodel.SinglePhaseFlowTV(
    kernel=kernel_obj, eos=eos, nlist=nlist,
    fluidgroup_filter=filterfluid, solidgroup_filter=filtersolid,
    densitymethod='SUMMATION')

model.mu               = viscosity
model.gx               = fx
model.damp             = 1000
model.artificialviscosity = True
model.alpha            = 0.2
model.beta             = 0.0
model.densitydiffusion = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c, cond = model.compute_speedofsound(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

if device.communicator.rank == 0:
    print(f'Speed of sound: {c:.4f} m/s  ({cond})')

sph_helper.update_min_c0(device, model, c,
                          mode='uref', lref=lref, uref=refvel, cfactor=10.0)

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')

# ─── Integrator (KickDriftKickTV) ────────────────────────────────────────────
integrator = hoomd.sph.Integrator(dt=dt)
kdktv = hoomd.sph.methods.KickDriftKickTV(filter=filterfluid,
                                           densitymethod='SUMMATION')
integrator.methods.append(kdktv)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Output ──────────────────────────────────────────────────────────────────
gsd_writer = hoomd.write.GSD(filename=dumpname,
                              trigger=hoomd.trigger.Periodic(100),
                              mode='wb',
                              dynamic=['property', 'momentum'])
sim.operations.writers.append(gsd_writer)

logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps', 'walltime'])
compute_fluid = hoomd.sph.compute.SinglePhaseFlowBasicProperties(filter=filterfluid)
sim.operations.computes.append(compute_fluid)
logger.add(compute_fluid, quantities=['e_kin_fluid', 'mean_density'])
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(100), logger=logger,
                          max_header_len=10)
sim.operations.writers.append(table)

log_file = open(logname, mode='w+', newline='\n')
table_file = hoomd.write.Table(output=log_file,
                                trigger=hoomd.trigger.Periodic(100),
                                logger=logger, max_header_len=10)
sim.operations.writers.append(table_file)

# ─── Run ─────────────────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    print(f'Starting TV square-duct channel flow run at {dt_string}')
sim.run(steps, write_at_start=True)
gsd_writer.flush()

# ─── Post-processing: centreline velocity vs analytical ──────────────────────
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap = traj[-1]
        pos = snap.particles.position
        tid = snap.particles.typeid
        vel = snap.particles.velocity

    fluid = (tid == 0)
    y_f   = pos[fluid, 1]
    z_f   = pos[fluid, 2]
    vx_f  = vel[fluid, 0]

    near_centre = (np.abs(y_f) < 2.0 * dx) & (np.abs(z_f) < 2.0 * dx)
    v_cl_sph = float(np.mean(vx_f[near_centre])) if near_centre.any() else float(np.max(vx_f))

    print(f'\n── TV square-duct channel flow check (last frame, step {snap.configuration.step}) ──')
    print(f'  Analytical v_cl (square duct) = {v_cl_an:.5f} m/s')
    print(f'  SPH        v_cl               = {v_cl_sph:.5f} m/s')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_spf(dumpname)
