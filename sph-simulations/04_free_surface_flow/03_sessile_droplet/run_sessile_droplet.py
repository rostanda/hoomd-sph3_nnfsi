#!/usr/bin/env python3
r"""
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

Sessile droplet contact-angle benchmark — SinglePhaseFlowFS.

BENCHMARK DESCRIPTION
---------------------
A 2-D liquid droplet is initialised as a semicircle ($\theta_\mathrm{init} = 90°$) on a
flat solid wall.  Surface tension $\sigma$ and contact-angle enforcement
(Huber et al. 2016) drive the droplet toward the prescribed equilibrium
contact angle $\theta_\mathrm{eq}$.

The equilibrium droplet shape is a circular cap.  Given the initial
semicircle area $A = \pi R_\mathrm{drop}^2 / 2$ and the prescribed angle $\theta_\mathrm{eq}$, the
equilibrium cap radius $R_\mathrm{cap}$ satisfies:
    $\pi R_\mathrm{drop}^2 / 2 = R_\mathrm{cap}^2 (\theta_\mathrm{eq} - \sin\theta_\mathrm{eq} \cos\theta_\mathrm{eq})$

The equilibrium height $h$ and base half-width $r$ are:
    $h = R_\mathrm{cap} (1 - \cos\theta_\mathrm{eq})$
    $r = R_\mathrm{cap} \sin\theta_\mathrm{eq}$
    $\theta_\mathrm{eq} = 2 \arctan(h / r)$  — measured from the final particle distribution

Post-processing measures h and r from the last GSD frame and computes
the SPH contact angle, comparing it to the prescribed value.

IMPORTANT: sigma > 0 is required for the droplet to reach equilibrium;
set sigma = 0 to disable surface tension and keep only detection.

Usage:
    python3 run_sessile_droplet.py <num_length> <init_gsd_file> [steps] [theta_eq_deg]
    e.g.  python3 run_sessile_droplet.py 20 sessile_droplet_*_init.gsd 30001 60
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

num_length    = int(sys.argv[1])
filename      = str(sys.argv[2])
steps         = int(sys.argv[3])   if len(sys.argv) > 3 else 30001
theta_eq_deg  = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logname   = filename.replace('_init.gsd', f'_runFS_theta{int(theta_eq_deg)}.log')
dumpname  = filename.replace('_init.gsd', f'_runFS_theta{int(theta_eq_deg)}.gsd')

sim.create_state_from_gsd(filename=filename, domain_decomposition=(None, None, 1))

# ─── Physical parameters ─────────────────────────────────────────────────────
lref          = 0.001              # reference length                [m]
R_drop        = 0.4 * lref        # initial semicircle radius       [m]
dx            = R_drop / num_length
rho0          = 1000.0             # rest density                    [kg/m³]
# Viscosity chosen to give $Oh = \mu/\sqrt{\rho \sigma R} \approx 1.0$ (critically damped) so the
# contact line converges monotonically without growing oscillations.
# $Oh = \mu/\sqrt{\rho \sigma R} = \mu/\sqrt{1000 \times 0.072 \times 0.0004} = \mu/0.1697 = 1.0$ → $\mu = 0.1697\,\mathrm{Pa \cdot s}$
# Oh≈0.3 ($\mu$=0.051) caused underdamped parametric instability: growing oscillation
# amplitude over $O(10^4)$ steps. Oh≈1 damps in ~2.3 viscous times (30001 steps).
viscosity     = 0.170              # dynamic viscosity                [Pa·s]  (Oh≈1.0)
sigma         = 0.072              # surface tension coeff.           [N/m]
drho          = 0.01
backpress     = 0.01
nu            = viscosity / rho0
theta_eq      = np.radians(theta_eq_deg)

# Capillary velocity scale $U_\mathrm{cap} = \sigma / (\rho_0 R_\mathrm{drop})$
U_cap  = sigma / (rho0 * R_drop)
refvel = U_cap

# ─── Free-surface parameters ──────────────────────────────────────────────────
fs_threshold  = 0.99
contact_angle = theta_eq

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
model.gx                  = 0.0
model.gy                  = 0.0     # no gravity — surface tension drives shape
model.gz                  = 0.0
model.damp                = 1000    # ramps body force (=0 here, no gravity); kept for API completeness
model.artificialviscosity = True
model.alpha               = 0.5    # AV damping; physical viscosity (Oh≈0.3) already provides main damping
model.beta                = 0.0
model.densitydiffusion    = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

# For capillary flows: $c \gg \sqrt{\sigma / (\rho_0 R_\mathrm{drop})}$
# Use $c \geq 10 \sqrt{\sigma / (\rho_0 R_\mathrm{drop})}$ as lower bound
c_cap   = np.sqrt(sigma / (rho0 * R_drop))  # capillary wave speed scale
c_min   = max(10.0 * c_cap, 10.0 * refvel)
c       = c_min
eos.set_speedofsound(c)

if device.communicator.rank == 0:
    print(f'Capillary velocity U_cap = {U_cap:.4f} m/s')
    print(f'Speed of sound set to c = {c:.4f} m/s')
    print(f'Prescribed contact angle θ_eq = {theta_eq_deg:.1f}°')

dt_CFL     = 0.25 * dx / c
dt_visc    = dx**2 * rho0 / (8.0 * viscosity)
dt_capill  = np.sqrt(rho0 * dx**3 / (2.0 * np.pi * sigma))  # capillary timestep
dt         = 0.25 * min(dt_CFL, dt_visc, dt_capill)

if device.communicator.rank == 0:
    print(f'dt = {dt:.3e} s  (CFL: {dt_CFL:.3e}, visc: {dt_visc:.3e}, cap: {dt_capill:.3e})')
    t_end   = steps * dt
    t_visc  = rho0 * R_drop**2 / viscosity
    print(f'Simulated time: {t_end:.3e} s  (viscous scale t_visc = {t_visc:.3e} s)')

# ─── Integrator (KickDriftKickTV — required for SinglePhaseFlowFS) ────────────
integrator = hoomd.sph.Integrator(dt=dt)
kdktv = hoomd.sph.methods.KickDriftKickTV(filter=filterfluid,
                                           densitymethod='SUMMATION')
integrator.methods.append(kdktv)
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
                              trigger=hoomd.trigger.Periodic(1000),
                              mode='wb',
                              dynamic=['property', 'momentum'])
sim.operations.writers.append(gsd_writer)

logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps', 'walltime'])
compute_fluid = hoomd.sph.compute.SinglePhaseFlowBasicProperties(filter=filterfluid)
sim.operations.computes.append(compute_fluid)
logger.add(compute_fluid, quantities=['e_kin_fluid', 'mean_density'])
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(500), logger=logger,
                          max_header_len=10)
sim.operations.writers.append(table)

log_file = open(logname, mode='w+', newline='\n')
table_file = hoomd.write.Table(output=log_file,
                                trigger=hoomd.trigger.Periodic(500),
                                logger=logger, max_header_len=10)
sim.operations.writers.append(table_file)

# ─── Apply safety limiters after attachment ───────────────────────────────────
# Trigger prepRun to attach all operations (integrator + writers).
# Then set velocity and displacement limiters directly on the C++ object.
# These prevent isolated surface particles from developing runaway positions
# when surface-tension forces are large during the initial transient.
sim.run(0, write_at_start=False)
kdktv.setvLimit(2.0 * c)    # cap per-component velocity at 2x speed of sound
kdktv.setxLimit(0.5 * dx)   # cap displacement per step at 0.5 particle spacings

# ─── Run ─────────────────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    print(f'Starting sessile-droplet run (θ_eq = {theta_eq_deg:.0f}°) at {dt_string}')
sim.run(steps, write_at_start=True)
gsd_writer.flush()

# ─── Post-processing: measure contact angle from final droplet shape ──────────
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap = traj[-1]
        pos  = snap.particles.position
        tid  = snap.particles.typeid
        vel  = snap.particles.velocity

    fluid  = (tid == 0)
    x_f    = pos[fluid, 0]
    y_f    = pos[fluid, 1]
    vel_f  = vel[fluid, :2]

    # Floor y level: minimum y of fluid particles + 0.5*dx
    y_floor_est = float(np.min(y_f)) + 0.5 * dx

    # Droplet height h: maximum y above the floor
    h = float(np.max(y_f)) - y_floor_est

    # Base half-width r: half the x-extent of the lowest fluid layer
    base_layer  = y_f < (y_floor_est + 1.5 * dx)
    if np.any(base_layer):
        r = (float(np.max(x_f[base_layer])) - float(np.min(x_f[base_layer]))) / 2.0
    else:
        r = float('nan')

    # Contact angle from spherical-cap geometry: $\theta = 2 \arctan(h / r)$
    if r > 0:
        theta_sph     = 2.0 * np.arctan(h / r)
        theta_sph_deg = np.degrees(theta_sph)
        theta_err_deg = abs(theta_sph_deg - theta_eq_deg)
    else:
        theta_sph_deg = float('nan')
        theta_err_deg = float('nan')

    # Analytical equilibrium cap radius from volume conservation:
    # $A_\mathrm{semicircle} = \pi R^2 / 2$;  $A_\mathrm{cap} = R_\mathrm{cap}^2 (\theta - \sin\theta \cos\theta)$
    A_semi    = np.pi * R_drop**2 / 2.0
    denom     = theta_eq - np.sin(theta_eq) * np.cos(theta_eq)
    R_cap_an  = np.sqrt(A_semi / denom) if denom > 0 else float('nan')
    h_an      = R_cap_an * (1.0 - np.cos(theta_eq))
    r_an      = R_cap_an * np.sin(theta_eq)

    v_rms = float(np.sqrt(np.mean(np.sum(vel_f**2, axis=1))))

    print(f'\n── Sessile droplet check (last frame, step {snap.configuration.step}) ──')
    print(f'  Prescribed θ_eq     = {theta_eq_deg:.1f}°')
    print(f'  SPH θ (from h,r)    = {theta_sph_deg:.1f}°  (error: {theta_err_deg:.1f}°)')
    print(f'  SPH  h = {h*1e6:.1f} µm,  r = {r*1e6:.1f} µm')
    print(f'  Theor h = {h_an*1e6:.1f} µm,  r = {r_an*1e6:.1f} µm')
    print(f'  RMS velocity (should→0 at equilibrium) = {v_rms:.3e} m/s')
    print(f'  σ = {sigma} N/m,  R_drop_init = {R_drop*1e6:.0f} µm')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_fs(dumpname)
