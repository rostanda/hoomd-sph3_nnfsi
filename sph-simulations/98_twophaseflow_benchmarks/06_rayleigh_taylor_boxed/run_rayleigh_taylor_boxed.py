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

Rayleigh–Taylor instability (boxed) — WCSPH run script.

BENCHMARK DESCRIPTION
---------------------
A heavy fluid (ρ₁ = 1500 kg/m³) rests above a light fluid (ρ₂ = 500 kg/m³)
in a domain with solid walls in x and y and periodic boundaries in z.
Gravity acts downward.  The interface is perturbed sinusoidally
(δ = 0.01 lref) to seed the dominant mode.  Surface tension (σ = 0.01 N/m)
stabilises small-scale noise.

Compared to 05_rayleigh_taylor (periodic in x), the lateral solid walls
confine the flow and suppress cross-domain periodicity artefacts.

Atwood number: At = (ρ₁ − ρ₂) / (ρ₁ + ρ₂) = 0.5

Linear growth (inviscid):  γ ≈ 176 s⁻¹
→ for δ = 10 µm, amplitude reaches lref/10 after t ≈ ln(10)/γ ≈ 13 ms.

Usage:
    python3 run_rayleigh_taylor_boxed.py <num_length> <init_gsd_file> [steps]
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

sim.create_state_from_gsd(filename=filename, domain_decomposition=(None, None, 1))

# ─── Physical parameters ─────────────────────────────────────────────────────
lref       = 0.001        # reference length (domain width)   [m]
dx         = lref / num_length
rho01      = 1500.0       # rest density heavy fluid 'W'      [kg/m³]
rho02      = 500.0        # rest density light fluid 'N'      [kg/m³]
viscosity1 = 0.002        # dynamic viscosity heavy fluid     [Pa·s]
viscosity2 = 0.002        # dynamic viscosity light fluid     [Pa·s]
sigma      = 0.01         # surface tension                   [N/m]
gy         = -9.81        # gravitational acceleration        [m/s²]
backpress  = 0.01         # background pressure coeff         [–]
drho       = 0.01         # allowed density variation         [–]
steps      = int(sys.argv[3]) if len(sys.argv) > 3 else 50001  # simulation steps

At     = (rho01 - rho02) / (rho01 + rho02)
g_mag  = abs(gy)
k_mode = 2 * np.pi / lref
gamma  = np.sqrt(At * g_mag * k_mode)   # inviscid linear growth rate

# Reference velocity from linear growth over one lref
U_ref = gamma * lref * 0.01            # 1% of lref at growth rate

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
filterfluidW = hoomd.filter.Type(['W'])   # heavy fluid (phase 1)
filterfluidN = hoomd.filter.Type(['N'])   # light fluid (phase 2)
filtersolid  = hoomd.filter.Type(['S'])   # solid walls (x-left/right + y-top/bottom)

# ─── TwoPhaseFlow model ──────────────────────────────────────────────────────
model = hoomd.sph.sphmodel.TwoPhaseFlow(
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
model.omega            = 90          # neutral wetting at solid walls
model.gy               = gy          # gravity in y direction
model.damp             = 0           # no artificial damping for RT
model.artificialviscosity = True
model.alpha            = 0.05        # low AV to allow instability to grow
model.beta             = 0.0
model.densitydiffusion = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c1, cond1, c2, cond2 = model.compute_speedofsound(
    LREF=lref, UREF=U_ref, DX=dx, DRHO=drho,
    H=maximum_smoothing_length,
    MU1=viscosity1, MU2=viscosity2,
    RHO01=rho01, RHO02=rho02, SIGMA12=sigma)

if device.communicator.rank == 0:
    print(f'Phase W speed of sound: {c1:.4f} m/s  ({cond1})')
    print(f'Phase N speed of sound: {c2:.4f} m/s  ({cond2})')

sph_helper.update_min_c0_tpf(device, model, c1, c2,
                              mode='plain', lref=lref, uref=U_ref, cfactor=10.0)

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=U_ref, DX=dx, DRHO=drho,
    H=maximum_smoothing_length,
    MU1=viscosity1, MU2=viscosity2,
    RHO01=rho01, RHO02=rho02, SIGMA12=sigma)

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')
    print(f'Atwood number    At = {At:.3f}')
    print(f'Linear growth    γ  = {gamma:.1f} s⁻¹')

# ─── Integrator ──────────────────────────────────────────────────────────────
integrator = hoomd.sph.Integrator(dt=dt)
vvbW = hoomd.sph.methods.VelocityVerletBasic(filter=filterfluidW,
                                              densitymethod='SUMMATION')
vvbN = hoomd.sph.methods.VelocityVerletBasic(filter=filterfluidN,
                                              densitymethod='SUMMATION')
integrator.methods.append(vvbW)
integrator.methods.append(vvbN)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Output ──────────────────────────────────────────────────────────────────
gsd_writer = hoomd.write.GSD(filename=dumpname,
                              trigger=hoomd.trigger.Periodic(500),
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
    print(f'Starting WCSPH Rayleigh–Taylor (boxed) run at {dt_string}')
sim.run(steps, write_at_start=True)
gsd_writer.flush()

# ─── Post-processing: bubble and spike tip tracking ──────────────────────────
if device.communicator.rank == 0:
    delta = 0.01 * lref  # initial perturbation amplitude

    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap_last = traj[-1]
        tid = snap_last.particles.typeid
        pos = snap_last.particles.position

    light = (tid == 1)   # 'N' — light fluid
    heavy = (tid == 0)   # 'W' — heavy fluid

    # Bubble tip: highest y among light-fluid particles in centre column
    # (x ≈ 0 where the sinusoidal perturbation peaks upward for light fluid)
    centre_x = np.abs(pos[:, 0]) < lref / 4
    if np.any(light & centre_x):
        y_bubble = float(np.max(pos[light & centre_x, 1]))
    else:
        y_bubble = float(np.max(pos[light, 1]))

    # Spike tip: lowest y among heavy-fluid particles in edge column
    # (x ≈ ±lref/2 where the heavy fluid penetrates downward)
    edge_x = np.abs(pos[:, 0]) > lref / 4
    if np.any(heavy & edge_x):
        y_spike = float(np.min(pos[heavy & edge_x, 1]))
    else:
        y_spike = float(np.min(pos[heavy, 1]))

    t_sim = snap_last.configuration.step * dt
    eta_lin = delta * np.exp(gamma * t_sim)  # linear theory amplitude

    print(f'\n── Rayleigh–Taylor (boxed) summary (last frame, step {snap_last.configuration.step}) ──')
    print(f'  Simulation time  t  = {t_sim*1e3:.2f} ms')
    print(f'  Bubble tip       y  = {y_bubble*1e3:.3f} mm')
    print(f'  Spike  tip       y  = {y_spike*1e3:.3f} mm')
    print(f'  Linear theory amplitude (inviscid) = {eta_lin*1e3:.3f} mm')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_tpf(dumpname)
