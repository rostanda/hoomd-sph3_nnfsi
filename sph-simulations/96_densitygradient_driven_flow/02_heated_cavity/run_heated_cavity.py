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

Differentially heated square cavity — SinglePhaseFlowGDGD (Boussinesq) run script.

BENCHMARK DESCRIPTION
---------------------
Square cavity (L = 1 m) with:
  Left wall  (x = −L/2): hot,  T_hot  = 1.0
  Right wall (x = +L/2): cold, T_cold = 0.0
  Top/bottom walls:       T_avg = 0.5  (approximate adiabatic condition)

Gravity acts in the −y direction.  The Boussinesq approximation applies a
per-particle buoyancy correction:
    ΔF_b = m × g × (−β × (T_i − T_cold))
so hot fluid near the left wall experiences an upward buoyancy force and
rises, cold fluid near the right wall sinks.  A stable circulation cell forms.

Non-dimensional parameters (chosen for numerical convenience):
    L      = 1.0 m        cavity side length
    ρ₀     = 1.0 kg/m³   rest density
    g      = 1.0 m/s²    gravitational acceleration
    ΔT     = 1.0 K        temperature difference (T_hot − T_cold)
    β      = 0.001 1/K   thermal expansion coefficient
    ν = μ  = 0.001 m²/s  kinematic viscosity (ρ₀ = 1 → μ = ν)
    κ_s    = 0.001 m²/s  thermal diffusivity

Derived dimensionless numbers:
    Ra = g β ΔT L³ / (ν κ_s) = 1.0 × 0.001 × 1.0 × 1³ / (0.001 × 0.001) = 1000
    Pr = ν / κ_s = 1.0

Reference Nusselt number (de Vahl Davis 1983):
    Nu(Ra=1000) ≈ 1.118

Note on adiabatic walls: True adiabatic (zero-flux) top/bottom conditions
require updating solid particle temperatures each step.  Here the simpler
approximation T_top/bottom = T_avg = 0.5 is used, which introduces a small
error at the corners but keeps the script self-contained.

Usage:
    python3 run_heated_cavity.py <num_length> <init_gsd_file> [steps]

    num_length    : integer resolution (particles across L, e.g. 20, 40)
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
L          = 1.0         # cavity side length                     [m]
dx         = L / num_length
rho0       = 1.0         # rest density                           [kg/m³]
viscosity  = 0.001       # dynamic viscosity  (= ν since ρ₀=1)   [Pa·s]
kappa_s    = 0.001       # thermal diffusivity (Pr = 1)           [m²/s]
g          = 1.0         # gravitational acceleration             [m/s²]
beta_s     = 0.001       # thermal expansion coefficient          [1/K]
T_cold     = 0.0         # cold-wall (right) temperature
T_hot      = 1.0         # hot-wall  (left)  temperature
T_avg      = 0.5 * (T_hot + T_cold)   # top/bottom wall (adiabatic approx)
DeltaT     = T_hot - T_cold

nu  = viscosity / rho0
Ra  = g * beta_s * DeltaT * L**3 / (nu * kappa_s)
Pr  = nu / kappa_s

drho      = 0.01         # allowed density variation              [–]
backpress = 0.01         # background pressure coefficient        [–]

# Reference velocity: buoyancy velocity scale sqrt(g β ΔT L)
refvel = np.sqrt(g * beta_s * DeltaT * L)

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

# ─── SinglePhaseFlowGDGD model (Boussinesq natural convection) ───────────────
# Boussinesq mode: standard EOS pressure, explicit per-particle buoyancy:
#   ΔF_b = m × g_y × (−β × (T_i − T_cold))
# With g_y = −g and T_cold as reference, hot particles (T > 0) receive an
# upward force, driving the convection roll.
model = hoomd.sph.sphmodel.SinglePhaseFlowGDGD(
    kernel=kernel_obj, eos=eos, nlist=nlist,
    fluidgroup_filter=filterfluid, solidgroup_filter=filtersolid,
    densitymethod='SUMMATION',
    kappa_s=kappa_s,
    beta_s=beta_s,
    scalar_ref=T_cold,   # Boussinesq reference: F_b = 0 at T = T_cold
    boussinesq=True,
)

model.mu                  = viscosity
model.gx                  = 0.0    # no horizontal body force
model.gy                  = -g     # gravity acts downward (−y)
model.gz                  = 0.0
model.damp                = 1000
model.artificialviscosity = True
model.alpha               = 0.2
model.beta                = 0.0
model.densitydiffusion    = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c, cond = model.compute_speedofsound(
    LREF=L, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

eos.set_speedofsound(c)   # required so compute_dt uses the correct c

if device.communicator.rank == 0:
    print(f'Ra = {Ra:.0f},  Pr = {Pr:.2f}')
    print(f'U_ref = sqrt(g β ΔT L) = {refvel:.4f} m/s')
    print(f'Speed of sound: {c:.4f} m/s  ({cond})')

dt, dt_cond = model.compute_dt(
    LREF=L, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

tau_flow = L / refvel   # advective time scale [s]

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')
    print(f'τ_flow = L/U_ref = {tau_flow:.1f} s  ≈ {int(tau_flow/dt)} steps')

# ─── Integrator ──────────────────────────────────────────────────────────────
integrator = hoomd.sph.Integrator(dt=dt)
# VelocityVerletBasic advances aux4.x (scalar T) via the dpedt.z half-step.
vvb = hoomd.sph.methods.VelocityVerletBasic(filter=filterfluid,
                                              densitymethod='SUMMATION')
integrator.methods.append(vvb)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Initialise scalar field T (aux4.x) ──────────────────────────────────────
# Left  solid (x < −L/2): T_hot.
# Right solid (x >  L/2): T_cold.
# Top/bottom solid: T_avg (adiabatic approximation).
# Fluid: linear initial temperature T = T_hot + (T_cold−T_hot)*(x+L/2)/L
#        (aligned with steady-state profile; improves convergence speed).
with sim.state.cpu_local_snapshot as snap:
    pos  = snap.particles.position[:]
    tid  = snap.particles.typeid[:]
    aux4 = snap.particles.auxiliary4   # (N_local, 3); T = column 0

    x_pos = pos[:, 0]

    is_fluid = (tid == 0)
    is_solid = (tid == 1)
    is_left  = is_solid & (x_pos < -0.5 * L)   # hot wall
    is_right = is_solid & (x_pos >  0.5 * L)   # cold wall
    is_tb    = is_solid & (~is_left) & (~is_right)   # top/bottom

    # Linear initial temperature field (helps convergence)
    aux4[is_fluid, 0] = T_hot + (T_cold - T_hot) * (x_pos[is_fluid] + 0.5 * L) / L
    aux4[is_left,  0] = T_hot
    aux4[is_right, 0] = T_cold
    aux4[is_tb,    0] = T_avg

# ─── Output ──────────────────────────────────────────────────────────────────
gsd_writer = hoomd.write.GSD(filename=dumpname,
                              trigger=hoomd.trigger.Periodic(1000),
                              mode='wb',
                              dynamic=['property', 'momentum'])
sim.operations.writers.append(gsd_writer)

logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps', 'walltime'])
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
    print(f'Starting heated cavity run at {dt_string}')
    print(f'  Left (hot) wall:    T = {T_hot:.1f}  |  Right (cold) wall: T = {T_cold:.1f}')
    print(f'  Top/bottom walls:   T = {T_avg:.2f}  (approximate adiabatic)')

sim.run(steps, write_at_start=True)

# ─── Post-processing ─────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap = traj[-1]
        pos  = snap.particles.position
        tid  = snap.particles.typeid
        vel  = snap.particles.velocity
        aux4 = snap.particles.aux4   # shape (N, 3); T = aux4[:, 0]

    fluid = (tid == 0)
    x_f   = pos[fluid, 0]
    y_f   = pos[fluid, 1]
    vx_f  = vel[fluid, 0]
    vy_f  = vel[fluid, 1]
    T_f   = aux4[fluid, 0]

    # Maximum velocity in the domain (indicator of convection strength)
    v_mag = np.sqrt(vx_f**2 + vy_f**2)
    v_max = float(np.max(v_mag))

    # Nusselt number estimate at the hot wall (x ≈ −L/2)
    # Nu = L / ΔT × mean(dT/dx) at x = −L/2
    # Approximate dT/dx using the first fluid layer adjacent to the left wall.
    x_thresh = -0.5 * L + 2.5 * dx
    near_hot  = (x_f < x_thresh) & (np.abs(y_f) < 0.5 * L)
    if np.sum(near_hot) > 0:
        x_near   = x_f[near_hot]
        T_near   = T_f[near_hot]
        # Finite-difference gradient (hot wall is at x = −L/2, T = T_hot)
        dTdx_hot = np.mean((T_near - T_hot) / (x_near - (-0.5 * L)))
        Nu_hot   = -L / DeltaT * dTdx_hot
    else:
        Nu_hot = float('nan')

    # Reference Nu values from de Vahl Davis (1983)
    Nu_ref = {1000: 1.118, 10000: 2.243, 100000: 4.519, 1000000: 8.800}

    print(f'\n── Heated cavity check (last frame, step {snap.configuration.step}) ──')
    print(f'  Ra = {Ra:.0f},  Pr = {Pr:.2f}')
    print(f'  Max fluid velocity = {v_max:.4e} m/s  (U_ref = {refvel:.4e} m/s)')
    print(f'  Nu (hot-wall estimate) = {Nu_hot:.3f}')
    Ra_key = int(round(Ra))
    if Ra_key in Nu_ref:
        Nu_dvd = Nu_ref[Ra_key]
        err    = abs(Nu_hot - Nu_dvd) / Nu_dvd * 100.0
        print(f'  Nu reference (de Vahl Davis 1983) = {Nu_dvd:.3f}')
        print(f'  Relative error = {err:.1f} %')
    else:
        print(f'  (No reference Nu for Ra = {Ra:.0e}; '
              f'see de Vahl Davis 1983 for Ra = 1e3…1e6)')

    # Temperature distribution check: left half should be hotter than right half
    left_half  = (x_f < 0)
    right_half = (x_f > 0)
    T_left  = float(np.mean(T_f[left_half]))
    T_right = float(np.mean(T_f[right_half]))
    print(f'  Mean T (left  half) = {T_left:.4f}  (expected > {T_avg:.2f})')
    print(f'  Mean T (right half) = {T_right:.4f}  (expected < {T_avg:.2f})')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_gdgd(dumpname)
