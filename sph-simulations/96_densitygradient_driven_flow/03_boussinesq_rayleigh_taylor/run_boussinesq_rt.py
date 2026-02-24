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

Boussinesq Rayleigh–Taylor instability — SinglePhaseFlowGDGD run script.

BENCHMARK DESCRIPTION
---------------------
A layer of heavy cold fluid (T = 0) rests on top of a layer of light hot fluid
(T = 1), separated by a sinusoidally perturbed interface:

    y_int(x) = δ · cos(2π·x/lref),   δ = 0.1·lref

Gravity (gy = -9.81 m/s²) destabilises the interface, driving the classic
Rayleigh–Taylor instability.  Buoyancy is provided via the Boussinesq term in
SinglePhaseFlowGDGD:

    F_buoy = m · g · (−β_s · (T − T_ref))

    β_s = 0.5,  T_ref = 0.5,  ΔT = 1  →  At_eff = β_s · ΔT / 2 = 0.25

With kappa_s = 0 the scalar T is frozen to each Lagrangian particle (no
diffusion), giving a sharp-interface RT.  In the linear regime the interface
amplitude grows as:

    δ(t) = δ_0 · cosh(γ · t),   γ = sqrt(At_eff · |gy| · 2π/lref)

Physical parameters:
    lref      = 0.001 m        perturbation wavelength = cavity width
    rho0      = 1000 kg/m³     rest density
    viscosity = 0.002 Pa·s     dynamic viscosity
    gy        = -9.81 m/s²     gravity
    beta_s    = 0.5            Boussinesq coefficient
    kappa_s   = 0.0            no scalar diffusion (sharp interface)
    At_eff    = 0.25           effective Atwood number
    γ         ≈ 124 s⁻¹       linear growth rate  (for lref = 0.001 m)
    τ_lin     = 1/γ ≈ 8 ms

Usage:
    python3 run_boussinesq_rt.py <num_length> <init_gsd_file> [steps]

    num_length    : particles across the cavity width lref (integer, e.g. 20)
    init_gsd_file : GSD file produced by create_input_geometry.py
    steps         : simulation steps (default: 10001)
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
logname   = filename.replace('_init.gsd', '_run.log')
dumpname  = filename.replace('_init.gsd', '_run.gsd')

sim.create_state_from_gsd(filename=filename)

# ─── Physical parameters ─────────────────────────────────────────────────────
lref      = 0.001        # perturbation wavelength = cavity width     [m]
dx        = lref / num_length
rho0      = 1000.0       # rest density                               [kg/m³]
viscosity = 0.002        # dynamic viscosity                          [Pa·s]
gy        = -9.81        # gravity (y direction, downward)            [m/s²]
beta_s    = 0.5          # Boussinesq thermal expansion coefficient   [–]
kappa_s   = 0.0          # no scalar diffusion → sharp interface      [m²/s]
T_hot     = 1.0          # lower-half temperature (light, rises)      [–]
T_cold    = 0.0          # upper-half temperature (heavy, sinks)      [–]
T_ref     = 0.5 * (T_hot + T_cold)
DeltaT    = T_hot - T_cold
delta     = 0.1 * lref   # initial interface perturbation amplitude   [m]

drho      = 0.01         # allowed density variation                  [–]
backpress = 0.01         # background pressure coefficient            [–]

At_eff    = beta_s * DeltaT / 2          # effective Atwood number  = 0.25
k_wave    = 2.0 * np.pi / lref           # perturbation wavenumber  [1/m]
gamma_lin = np.sqrt(At_eff * abs(gy) * k_wave)   # linear growth rate [1/s]
refvel    = np.sqrt(At_eff * abs(gy) * lref)      # buoyancy velocity scale [m/s]

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

# ─── SinglePhaseFlowGDGD model ───────────────────────────────────────────────
# Boussinesq mode: buoyancy = m·g·(−β_s·(T−T_ref)).
# kappa_s = 0 → T is frozen (Lagrangian advection only, sharp interface).
# beta_s = 0.5, T ∈ {0, 1} → At_eff = 0.25.
model = hoomd.sph.sphmodel.SinglePhaseFlowGDGD(
    kernel=kernel_obj, eos=eos, nlist=nlist,
    fluidgroup_filter=filterfluid, solidgroup_filter=filtersolid,
    densitymethod='SUMMATION',
    kappa_s=kappa_s,
    beta_s=beta_s,
    scalar_ref=T_ref,
    boussinesq=True,
)

model.mu                  = viscosity
model.gx                  = 0.0
model.gy                  = gy
model.gz                  = 0.0
model.damp                = 1      # gravity ramps in over 1 step (effectively immediate)
model.artificialviscosity = True
model.alpha               = 0.1
model.beta                = 0.0
model.densitydiffusion    = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c, cond = model.compute_speedofsound(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

eos.set_speedofsound(c)

if device.communicator.rank == 0:
    print(f'Speed of sound: {c:.4f} m/s  ({cond})')
    print(f'At_eff = {At_eff:.3f},  γ_lin = {gamma_lin:.1f} s⁻¹,  '
          f'τ_lin = {1.0/gamma_lin*1e3:.2f} ms')

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

tau_lin   = 1.0 / gamma_lin
steps_lin = int(tau_lin / dt)

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')
    print(f'τ_lin = 1/γ = {tau_lin*1e3:.2f} ms  ≈ {steps_lin} steps')

# ─── Integrator ──────────────────────────────────────────────────────────────
integrator = hoomd.sph.Integrator(dt=dt)
# VelocityVerletBasic advances aux4.x (scalar T) via the dpedt.z half-step.
# With kappa_s = 0 the rate is zero and T stays exactly at its initial value.
vvb = hoomd.sph.methods.VelocityVerletBasic(filter=filterfluid,
                                              densitymethod='SUMMATION')
integrator.methods.append(vvb)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Initialise scalar field T (aux4.x) via perturbed interface ──────────────
# Interface: y_int(x) = δ · cos(2π·x/lref)
# Particles above interface (y > y_int) → cold (T = 0, heavy, will sink)
# Particles below interface (y ≤ y_int) → hot  (T = 1, light, will rise)
# Solid particles             → T_ref (neutral, no diffusion anyway)
with sim.state.cpu_local_snapshot as snap:
    pos  = snap.particles.position[:]   # (N_local, 3)
    tid  = snap.particles.typeid[:]     # (N_local,)
    aux4 = snap.particles.auxiliary4    # (N_local, 3); T stored in column 0 (x)

    x_p   = pos[:, 0]
    y_p   = pos[:, 1]
    y_int = delta * np.cos(2.0 * np.pi * x_p / lref)

    is_fluid = (tid == 0)
    is_upper = is_fluid & (y_p >  y_int)   # heavy cold (sinks)
    is_lower = is_fluid & (y_p <= y_int)   # light hot  (rises)

    aux4[:, 0]        = T_ref    # all particles: initialise to T_ref
    aux4[is_upper, 0] = T_cold   # upper fluid: cold, heavy
    aux4[is_lower, 0] = T_hot    # lower fluid: hot, light
    # solid particles retain T_ref (neutral; no diffusion across walls)

# ─── Output ──────────────────────────────────────────────────────────────────
gsd_writer = hoomd.write.GSD(filename=dumpname,
                              trigger=hoomd.trigger.Periodic(2000),
                              mode='wb',
                              dynamic=['property', 'momentum'])
sim.operations.writers.append(gsd_writer)

logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps', 'walltime'])
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(2000), logger=logger,
                          max_header_len=10)
sim.operations.writers.append(table)

log_file   = open(logname, mode='w+', newline='\n')
table_file = hoomd.write.Table(output=log_file,
                                trigger=hoomd.trigger.Periodic(2000),
                                logger=logger, max_header_len=10)
sim.operations.writers.append(table_file)

# ─── Run ─────────────────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    print(f'Starting Boussinesq RT run at {dt_string}')
    print(f'  Cold (T = {T_cold:.1f}) on top   → heavy, sinks  (At_eff = {At_eff:.3f})')
    print(f'  Hot  (T = {T_hot:.1f}) on bottom → light, rises')
    print(f'  Perturbation:  δ = {delta*1e6:.1f} µm  = {delta/dx:.1f} × dx')
    print(f'  Running {steps} steps  ({steps / steps_lin:.1f} × τ_lin)')

sim.run(steps, write_at_start=True)

# ─── Post-processing: bubble/spike tip vs linear theory ──────────────────────
# Since kappa_s = 0, T is frozen to each particle for all time.
# aux4[:, 0] read from GSD frame 0 gives the permanent T identity of each particle.
# Positions come from each frame.  The GSD reader falls back to the most-recently
# written frame for non-dynamic arrays (aux4), so snap.particles.aux4 always
# returns the correct T values.
if device.communicator.rank == 0:
    frames_step  = []
    frames_bub   = []   # max y of rising hot fluid (bubble front)
    frames_spike = []   # min y of sinking cold fluid (spike front)

    with gsd.hoomd.open(dumpname, 'r') as traj:
        for snap in traj:
            t_step = snap.configuration.step
            pos_s  = snap.particles.position
            tid_s  = snap.particles.typeid
            aux4_s = snap.particles.aux4   # shape (N, 3); T = column 0

            fluid    = (tid_s == 0)
            y_fl     = pos_s[fluid, 1]
            T_fl     = aux4_s[fluid, 0]

            # Bubble tip: max y of hot particles (T = T_hot) that have risen above y = 0
            hot_above = (T_fl > T_ref) & (y_fl > 0.0)
            bub_tip   = float(np.max(y_fl[hot_above])) if np.any(hot_above) else 0.0

            # Spike tip: min y of cold particles (T = T_cold) that have sunk below y = 0
            cold_below = (T_fl < T_ref) & (y_fl < 0.0)
            spike_tip  = float(np.min(y_fl[cold_below])) if np.any(cold_below) else 0.0

            frames_step.append(t_step)
            frames_bub.append(bub_tip)
            frames_spike.append(spike_tip)

    frames_step  = np.array(frames_step,  dtype=np.int64)
    frames_t     = frames_step * dt
    frames_bub   = np.array(frames_bub)
    frames_spike = np.array(frames_spike)

    # Measure actual initial amplitude from step-0 frame (discretisation shifts it slightly)
    delta_meas = 0.5 * (frames_bub[0] + abs(frames_spike[0]))
    amp_theory = delta_meas * np.cosh(gamma_lin * frames_t)

    print(f'\n── Boussinesq RT check ──')
    print(f'  γ_lin = {gamma_lin:.2f} s⁻¹  (At_eff = {At_eff:.3f},  lref = {lref*1e3:.1f} mm)')
    print(f'  Prescribed δ = {delta*1e6:.1f} µm,  measured δ₀ = {delta_meas*1e6:.2f} µm'
          f'  (discretisation: {abs(delta_meas-delta)/dx:.2f} dx)')
    print(f'  Linear regime ends at γ·t ≈ 2  →  t ≈ {2.0/gamma_lin*1e3:.1f} ms')
    print()
    print(f'  {"step":>8}  {"t [ms]":>8}  {"bubble [µm]":>12}  '
          f'{"spike [µm]":>11}  {"δ_theory [µm]":>14}  {"γ·t":>6}')
    for i in range(len(frames_t)):
        print(f'  {int(frames_step[i]):>8d}  {frames_t[i]*1e3:>8.3f}'
              f'  {frames_bub[i]*1e6:>12.2f}'
              f'  {frames_spike[i]*1e6:>11.2f}'
              f'  {amp_theory[i]*1e6:>14.2f}'
              f'  {gamma_lin*frames_t[i]:>6.2f}')

    # L₂ error of amplitude growth in the linear regime (γ·t < 2)
    lin_mask = (gamma_lin * frames_t) <= 2.0
    if np.sum(lin_mask) > 1:
        amp_sph = 0.5 * (frames_bub[lin_mask] + np.abs(frames_spike[lin_mask]))
        L2_amp  = np.sqrt(np.mean((amp_sph - amp_theory[lin_mask])**2)) / delta * 100.0
        print(f'\n  L₂ error (amplitude, linear regime γ·t ≤ 2): {L2_amp:.2f} % of δ')
    else:
        print(f'\n  (no frames in linear regime γ·t ≤ 2 — reduce steps or increase output freq.)')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_gdgd(dumpname)
