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

Thermal Couette flow — SinglePhaseFlowGDGD run script.

BENCHMARK DESCRIPTION
---------------------
Shear-driven flow between two isothermal parallel plates separated by H = lref.
  Bottom plate: stationary,  temperature T_cold = 0.
  Top plate:    moving at U_lid in x,  temperature T_hot = 1.

At steady state BOTH the velocity and temperature profiles are linear:
    v(y) = U_lid × (y + H/2) / H           (Couette velocity)
    T(y) = T_cold + (T_hot − T_cold) × (y + H/2) / H  (Fourier conduction)

This benchmark tests the scalar diffusion operator of SinglePhaseFlowGDGD
in isolation (beta_s = 0, so no buoyancy effect).

Physical parameters:
    H       = 1 mm       channel gap
    U_lid   = 0.01 m/s   top-wall velocity
    rho0    = 1000 kg/m³ rest density
    mu      = 0.01 Pa·s  dynamic viscosity
    kappa_s = 1e-5 m²/s  thermal diffusivity
    Pr      = mu / (rho0 × kappa_s) = 1.0  (Prandtl number)

Diffusive time scale:  τ_diff = H² / kappa_s = 0.1 s

Usage:
    python3 run_thermal_couette.py <num_length> <init_gsd_file> [steps]

    num_length    : particles across the channel gap H (integer, e.g. 20)
    init_gsd_file : GSD file produced by create_input_geometry.py
    steps         : simulation steps (default: 20001)
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
steps      = int(sys.argv[3]) if len(sys.argv) > 3 else 20001

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logname   = filename.replace('_init.gsd', '_run.log')
dumpname  = filename.replace('_init.gsd', '_run.gsd')

sim.create_state_from_gsd(filename=filename)

# ─── Physical parameters ─────────────────────────────────────────────────────
lref      = 0.001        # channel gap                         [m]
H         = lref
dx        = lref / num_length
rho0      = 1000.0       # rest density                        [kg/m³]
viscosity = 0.01         # dynamic viscosity                   [Pa·s]
kappa_s   = 1.0e-5       # thermal diffusivity (Pr = 1.0)      [m²/s]
lidvel    = 0.01         # top-wall velocity (x direction)     [m/s]
T_cold    = 0.0          # bottom-wall temperature             [–]
T_hot     = 1.0          # top-wall temperature                [–]
T_ref     = 0.5 * (T_hot + T_cold)   # initial fluid temperature
DeltaT    = T_hot - T_cold

drho      = 0.01         # allowed density variation           [–]
backpress = 0.01         # background pressure coefficient     [–]
refvel    = lidvel       # reference velocity for c0 estimate

Re = rho0 * lidvel * H / viscosity
Pr = viscosity / (rho0 * kappa_s)

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

# ─── SinglePhaseFlowGDGD model (pure scalar diffusion, no buoyancy) ──────────
# beta_s = 0 disables all buoyancy effects; only the scalar diffusion term
# dT/dt += (kappa_s / V_i) * (V_i² + V_j²) * (T_i - T_j) * dW/dr / r
# is active.  The velocity field converges to the standard Couette profile
# while the temperature evolves to the linear Fourier solution.
model = hoomd.sph.sphmodel.SinglePhaseFlowGDGD(
    kernel=kernel_obj, eos=eos, nlist=nlist,
    fluidgroup_filter=filterfluid, solidgroup_filter=filtersolid,
    densitymethod='SUMMATION',
    kappa_s=kappa_s,
    beta_s=0.0,          # no buoyancy — isolates the scalar diffusion test
    scalar_ref=T_ref,
    boussinesq=True,     # Boussinesq mode; beta_s=0 → zero buoyancy correction,
                         # equivalent to VRD but uses standard EOS (avoids VRD path)
)

model.mu                  = viscosity
model.gx                  = 0.0   # no body force for Couette
model.damp                = 1000
model.artificialviscosity = True
model.alpha               = 0.2
model.beta                = 0.0
model.densitydiffusion    = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c, cond = model.compute_speedofsound(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

eos.set_speedofsound(c)   # must be called before compute_dt uses eos.SpeedOfSound

if device.communicator.rank == 0:
    print(f'Speed of sound: {c:.4f} m/s  ({cond})')
    print(f'Re = {Re:.2f},  Pr = {Pr:.2f}')

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

tau_diff   = H**2 / kappa_s       # diffusion time scale [s]
steps_diff = int(tau_diff / dt)   # steps per τ_diff

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')
    print(f'τ_diff = H²/κ_s = {tau_diff:.3f} s  ≈ {steps_diff} steps')

# ─── Integrator ──────────────────────────────────────────────────────────────
integrator = hoomd.sph.Integrator(dt=dt)
# VelocityVerletBasic advances aux4.x (scalar T) via the dpedt.z half-step,
# which is set by SinglePhaseFlowGDGD::forcecomputation().
vvb = hoomd.sph.methods.VelocityVerletBasic(filter=filterfluid,
                                              densitymethod='SUMMATION')
integrator.methods.append(vvb)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Initialise scalar field T (aux4.x) ──────────────────────────────────────
# The GSD file has aux4 = 0 everywhere.  Assign wall temperatures here.
# Fluid is initialised to T_ref (uniform); the diffusion term will evolve it
# to the linear steady-state profile T(y) = (y + H/2) / H.
with sim.state.cpu_local_snapshot as snap:
    pos  = snap.particles.position[:]   # (N_local, 3)
    tid  = snap.particles.typeid[:]     # (N_local,)
    aux4 = snap.particles.auxiliary4    # (N_local, 3); T is column 0 (x component)

    is_solid = (tid == 1)
    top_wall = is_solid & (pos[:, 1] > 0)
    bot_wall = is_solid & (pos[:, 1] < 0)

    aux4[:, 0]         = T_ref    # fluid + unassigned → average temperature
    aux4[top_wall, 0]  = T_hot    # top wall: hot
    aux4[bot_wall, 0]  = T_cold   # bottom wall: cold

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

log_file = open(logname, mode='w+', newline='\n')
table_file = hoomd.write.Table(output=log_file,
                                trigger=hoomd.trigger.Periodic(2000),
                                logger=logger, max_header_len=10)
sim.operations.writers.append(table_file)

# ─── Run ─────────────────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    print(f'Starting thermal Couette run at {dt_string}')
    print(f'  Bottom wall: T = {T_cold:.1f}  |  Top wall: T = {T_hot:.1f}')
    print(f'  Fluid init:  T = {T_ref:.2f} (uniform, will diffuse to linear profile)')
    print(f'  Running {steps} steps ({steps / steps_diff:.1f} × τ_diff)')

sim.run(steps, write_at_start=True)

# ─── Post-processing: L₂ errors vs analytical steady-state profiles ──────────
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap = traj[-1]
        pos  = snap.particles.position
        tid  = snap.particles.typeid
        vel  = snap.particles.velocity
        aux4 = snap.particles.aux4   # shape (N, 3); T = aux4[:, 0]

    fluid  = (tid == 0)
    y_f    = pos[fluid, 1]
    vx_f   = vel[fluid, 0]
    T_f    = aux4[fluid, 0]

    # Evaluate only inside the fluid gap (exclude near-wall boundary)
    in_gap = np.abs(y_f) < H / 2
    y_ev   = y_f[in_gap]
    vx_ev  = vx_f[in_gap]
    T_ev   = T_f[in_gap]

    # Analytical steady-state profiles
    vx_an  = lidvel * (y_ev + H / 2) / H
    T_an   = T_cold + DeltaT * (y_ev + H / 2) / H

    L2_vel  = np.sqrt(np.mean((vx_ev - vx_an)**2)) / lidvel * 100.0
    L2_temp = np.sqrt(np.mean((T_ev  - T_an )**2)) / DeltaT  * 100.0

    # Interpolate to channel midpoint y = 0
    idx_sort = np.argsort(y_ev)
    vx_mid = float(np.interp(0.0, y_ev[idx_sort], vx_ev[idx_sort]))
    T_mid  = float(np.interp(0.0, y_ev[idx_sort], T_ev[idx_sort]))

    print(f'\n── Thermal Couette check (last frame, step {snap.configuration.step}) ──')
    print(f'  Velocity profile:')
    print(f'    vx at y=0   (SPH)    = {vx_mid:.5f} m/s')
    print(f'    vx at y=0   (theory) = {0.5 * lidvel:.5f} m/s')
    print(f'    L₂ error / U_lid     = {L2_vel:.2f} %')
    print(f'  Temperature profile:')
    print(f'    T  at y=0   (SPH)    = {T_mid:.4f}')
    print(f'    T  at y=0   (theory) = {0.5 * (T_hot + T_cold):.4f}')
    print(f'    L₂ error / ΔT        = {L2_temp:.2f} %')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_gdgd(dumpname)
