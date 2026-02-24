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

Static droplet — WCSPH run script.

BENCHMARK DESCRIPTION
---------------------
A spherical droplet of phase 'B' sits at rest in an equal-density outer fluid
('A') inside a fully periodic box.  After equilibration the pressure jump across
the interface should satisfy the Young–Laplace relation

    ΔP_theory = 2σ / R = 2 × 0.01 / 2.5e-4 = 80 Pa

This script uses the standard WCSPH VelocityVerletBasic integrator.  Spurious
(parasitic) currents are an unavoidable artefact of the discrete colour-gradient
surface-tension model; their magnitude relative to σ/μ characterises solver
quality.  Run run_static_droplet_TV.py for the transport-velocity variant, which
typically achieves lower spurious velocities.

Usage:
    python3 run_static_droplet.py <num_length> <init_gsd_file>

    num_length    : integer resolution (particles across lref, e.g. 20, 40)
    init_gsd_file : GSD file produced by create_input_geometry.py
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
logname   = filename.replace('_init.gsd', '_run.log')
dumpname  = filename.replace('_init.gsd', '_run.gsd')

sim.create_state_from_gsd(filename=filename)

# ─── Physical parameters ─────────────────────────────────────────────────────
lref         = 0.001        # reference length              [m]
R_drop       = 0.25 * lref  # droplet radius                [m]
dx           = lref / num_length
rho01        = 1000.0       # rest density phase A (outer)  [kg/m³]
rho02        = 1000.0       # rest density phase B (droplet)[kg/m³]
viscosity1   = 0.001        # dynamic viscosity phase A     [Pa·s]
viscosity2   = 0.001        # dynamic viscosity phase B     [Pa·s]
sigma        = 0.01         # surface tension               [N/m]
backpress    = 0.01         # background pressure coeff     [–]
refvel       = 0.0          # reference velocity            [m/s]
drho         = 0.01         # allowed density variation     [–]
steps        = 2001         # simulation steps

dP_theory = 2.0 * sigma / R_drop   # 80 Pa

# ─── Kernel ──────────────────────────────────────────────────────────────────
kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength
kernel_obj = hoomd.sph.kernel.Kernels[kernel]()
kappa      = kernel_obj.Kappa()

nlist = hoomd.nsearch.nlist.Cell(buffer=rcut * 0.05,
                                  rebuild_check_delay=1, kappa=kappa)

# ─── EOS (Tait) ──────────────────────────────────────────────────────────────
eos1 = hoomd.sph.eos.Tait()
eos2 = hoomd.sph.eos.Tait()
eos1.set_params(rho01, backpress)
eos2.set_params(rho02, backpress)

# ─── Particle filters ────────────────────────────────────────────────────────
# No solid walls in this fully-periodic benchmark.
filterfluidA = hoomd.filter.Type(['A'])   # outer fluid  (phase 1)
filterfluidB = hoomd.filter.Type(['B'])   # inner droplet (phase 2)
filtersolid  = hoomd.filter.Null()        # no solid particles

# ─── TwoPhaseFlow model ──────────────────────────────────────────────────────
model = hoomd.sph.sphmodel.TwoPhaseFlow(
    kernel=kernel_obj,
    eos1=eos1, eos2=eos2,
    nlist=nlist,
    fluidgroup1_filter=filterfluidA,
    fluidgroup2_filter=filterfluidB,
    solidgroup_filter=filtersolid,
    densitymethod='SUMMATION',
    colorgradientmethod='DENSITYRATIO',
)

model.mu1              = viscosity1
model.mu2              = viscosity2
model.sigma12          = sigma
model.omega            = 90          # contact angle [°] — unused without walls
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

if device.communicator.rank == 0:
    print(f'Phase A speed of sound: {c1:.4f} m/s  ({cond1})')
    print(f'Phase B speed of sound: {c2:.4f} m/s  ({cond2})')

sph_helper.update_min_c0_tpf(device, model, c1, c2,
                              mode='plain', lref=lref, uref=refvel, cfactor=10.0)

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length,
    MU1=viscosity1, MU2=viscosity2,
    RHO01=rho01, RHO02=rho02, SIGMA12=sigma)

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')
    print(f'Expected ΔP = 2σ/R = {dP_theory:.1f} Pa')

# ─── Integrator (standard WCSPH) ─────────────────────────────────────────────
integrator = hoomd.sph.Integrator(dt=dt)
vvbA = hoomd.sph.methods.VelocityVerletBasic(filter=filterfluidA,
                                              densitymethod='SUMMATION')
vvbB = hoomd.sph.methods.VelocityVerletBasic(filter=filterfluidB,
                                              densitymethod='SUMMATION')
integrator.methods.append(vvbA)
integrator.methods.append(vvbB)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Output ──────────────────────────────────────────────────────────────────
gsd_writer = hoomd.write.GSD(filename=dumpname,
                              trigger=hoomd.trigger.Periodic(10),
                              mode='wb',
                              dynamic=['property', 'momentum'])
sim.operations.writers.append(gsd_writer)

logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps', 'walltime'])
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
    print(f'Starting WCSPH static-droplet run at {dt_string}')
sim.run(steps, write_at_start=True)

# ─── Post-processing: Laplace pressure check ─────────────────────────────────
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap = traj[-1]
        pos  = snap.particles.position
        tid  = snap.particles.typeid    # 0 = 'A', 1 = 'B'
        pres = snap.particles.pressure

    inside  = (tid == 1)
    P_in    = float(np.mean(pres[inside]))
    P_out   = float(np.mean(pres[~inside]))
    dP_sim  = P_in - P_out
    rel_err = abs(dP_sim - dP_theory) / dP_theory * 100.0

    U_cap = sigma / viscosity1          # capillary velocity scale [m/s]
    vx    = snap.particles.velocity[:, 0]
    vy    = snap.particles.velocity[:, 1]
    vz    = snap.particles.velocity[:, 2]
    v_max = float(np.max(np.sqrt(vx**2 + vy**2 + vz**2)))

    print(f'\n── Laplace pressure check (last frame, step {snap.configuration.step}) ──')
    print(f'  P_inside   = {P_in:.2f} Pa')
    print(f'  P_outside  = {P_out:.2f} Pa')
    print(f'  ΔP_simul   = {dP_sim:.2f} Pa')
    print(f'  ΔP_theory  = {dP_theory:.2f} Pa')
    print(f'  Relative error = {rel_err:.1f} %')
    print(f'\n── Spurious velocity check ──')
    print(f'  U_cap = σ/μ = {U_cap:.2f} m/s')
    print(f'  v_max = {v_max:.4f} m/s  ({v_max/U_cap*100:.2f} % of U_cap)')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_tpf(dumpname)
