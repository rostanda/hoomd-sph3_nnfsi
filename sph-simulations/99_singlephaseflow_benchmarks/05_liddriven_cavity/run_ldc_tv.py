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

Lid-driven cavity — Transport-Velocity (TV) run script.

BENCHMARK DESCRIPTION
---------------------
A square cavity ($l_\mathrm{ref} \times l_\mathrm{ref} \times$ small depth) with solid walls on all four sides.
The top wall moves at $U_\mathrm{lid} = 1.0$ m/s in the x-direction.  The other three
walls are stationary.

This benchmark covers a range of Reynolds numbers ($Re = \rho U_\mathrm{lid} l_\mathrm{ref} / \mu$).
At steady state the velocity field shows a primary recirculation vortex.
Reference data: Ghia et al. (1982) for $Re = 100, 400, 1000, 3200, 5000$.

Usage (OptionParser):
    python3 run_ldc_tv.py -n <resolution> -S <init_gsd_file> -i <steps_x1000> -R <reynolds>
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
from optparse import OptionParser

try:
    import export_gsd2vtu
    HAS_VTU = True
except ImportError:
    HAS_VTU = False

# ─── Device & simulation ─────────────────────────────────────────────────────
device = hoomd.device.CPU(notice_level=2)
sim    = hoomd.Simulation(device=device)

parser = OptionParser()
parser.add_option("-n", "--resolution", type=int,   dest="resolution", default=100)
parser.add_option("-S", "--initgsd",    type=str,   dest="initgsd",    default=None)
parser.add_option("-i", "--steps",      type=int,   dest="steps",      default=200)
parser.add_option("-R", "--reynolds",   type=float, dest="reynolds",   default=100.0)
(options, _) = parser.parse_args()

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logname   = options.initgsd.replace('_init.gsd', '_runTV.log')
dumpname  = options.initgsd.replace('_init.gsd', '_runTV.gsd')

sim.create_state_from_gsd(filename=options.initgsd, domain_decomposition=(None, None, 1))

# ─── Physical parameters ─────────────────────────────────────────────────────
num_length = options.resolution
lref       = 1.0            # cavity side length                [m]
dx         = lref / num_length
rho0       = 1.0            # rest density                      [kg/m³]
lidvel     = 1.0            # lid velocity                      [m/s]
viscosity  = rho0 * lidvel * lref / options.reynolds  # [Pa·s]
drho       = 0.05           # allowed density variation         [–]
backpress  = 0.01           # background pressure coeff         [–]
refvel     = lidvel         # reference velocity                [m/s]
Re         = options.reynolds
steps      = options.steps * 1000 + 1

if device.communicator.rank == 0:
    print(f'LDC TV: Re = {Re:.0f},  μ = {viscosity:.4e} Pa·s')

# ─── Kernel ──────────────────────────────────────────────────────────────────
kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength
kernel_obj = hoomd.sph.kernel.Kernels[kernel]()
kappa      = kernel_obj.Kappa()

nlist = hoomd.nsearch.nlist.Cell(buffer=rcut * 0.05,
                                  rebuild_check_delay=1, kappa=kappa)

# ─── EOS (Linear — near-incompressible cavity flow) ──────────────────────────
eos = hoomd.sph.eos.Linear()
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
model.gx               = 0.0
model.damp             = 5000
model.artificialviscosity = True
model.alpha            = 0.4
model.beta             = 0.2
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
                              trigger=hoomd.trigger.Periodic(2000),
                              mode='wb',
                              dynamic=['property', 'momentum'])
sim.operations.writers.append(gsd_writer)

logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps', 'walltime'])
compute_fluid = hoomd.sph.compute.SinglePhaseFlowBasicProperties(filter=filterfluid)
sim.operations.computes.append(compute_fluid)
logger.add(compute_fluid, quantities=['e_kin_fluid', 'mean_density'])
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(1000), logger=logger,
                          max_header_len=10)
sim.operations.writers.append(table)

log_file = open(logname, mode='w+', newline='\n')
table_file = hoomd.write.Table(output=log_file,
                                trigger=hoomd.trigger.Periodic(1000),
                                logger=logger, max_header_len=10)
sim.operations.writers.append(table_file)

# ─── Run ─────────────────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    print(f'Starting TV lid-driven cavity run (Re={Re:.0f}) at {dt_string}')
sim.run(steps, write_at_start=True)
gsd_writer.flush()

# ─── Post-processing: centreline velocity ───────────────────────────────────
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap = traj[-1]
        pos = snap.particles.position
        tid = snap.particles.typeid
        vel = snap.particles.velocity

    fluid = (tid == 0)
    x_f   = pos[fluid, 0]
    y_f   = pos[fluid, 1]
    vx_f  = vel[fluid, 0]
    vy_f  = vel[fluid, 1]

    # Vertical centreline: $x \approx 0$, extract $v_y(y)$
    ctr_x = np.abs(x_f) < 2.0 * dx
    if ctr_x.any():
        y_ctr   = y_f[ctr_x]
        vy_ctr  = vy_f[ctr_x]
        order   = np.argsort(y_ctr)
        y_ctr   = y_ctr[order]
        vy_ctr  = vy_ctr[order]
        v_max_ctr = float(np.max(np.abs(vy_ctr)))
    else:
        v_max_ctr = float('nan')

    # Horizontal centreline: $y \approx 0$, extract $v_x(x)$
    ctr_y = np.abs(y_f) < 2.0 * dx
    if ctr_y.any():
        x_ctr   = x_f[ctr_y]
        vx_ctr  = vx_f[ctr_y]
        order   = np.argsort(x_ctr)
        x_ctr   = x_ctr[order]
        vx_ctr  = vx_ctr[order]
        v_max_hctr = float(np.max(np.abs(vx_ctr)))
    else:
        v_max_hctr = float('nan')

    print(f'\n── LDC TV summary (Re={Re:.0f}, last frame, step {snap.configuration.step}) ──')
    print(f'  |vy|_max on vertical centreline   = {v_max_ctr:.4f} m/s')
    print(f'  |vx|_max on horizontal centreline = {v_max_hctr:.4f} m/s')
    print(f'  (Compare to Ghia 1982 reference data for Re={Re:.0f})')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_spf(dumpname)
