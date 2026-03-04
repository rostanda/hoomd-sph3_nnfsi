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

H₂ Bubble in Brine Under Shear + Snap-back — three-phase WCSPH run script.

BENCHMARK DESCRIPTION
---------------------
Underground Hydrogen Storage (UHS) pore-scale benchmark.  A sessile H₂ bubble
is attached to the underside of a caprock (top solid wall) in brine.  The
simulation runs in three consecutive phases without restart:

Phase 1 — Relaxation
  A cubic patch of H₂ ('W') sits against the stationary top caprock wall,
  surrounded by brine ('N').  Surface tension (σ) and the prescribed contact
  angle (θ_eq, typically 40° — hydrophobic caprock in UHS conditions) reshape
  the cube into a spherical-cap bubble.  Both solid walls are stationary.
  Gravity acts in the −y direction; H₂ is buoyant (ρ_H2 ≈ 100 kg/m³ at
  130 bar) so it remains pressed against the caprock.

Phase 2 — Shear
  After relaxation the BOTTOM solid wall is set to velocity U_wall in the
  x direction (Couette-like driving).  The resulting shear flow deforms the
  bubble.  Contact-angle hysteresis (θ_rec, θ_adv) pins the contact line,
  so the bubble tilts but the contact area does not fully slide.  The extent
  of deformation is characterised by the capillary number Ca = μ_brine U_wall / σ.

Phase 3 — Snap-back
  The bottom wall velocity is reset to zero.  With hysteresis pinning active
  the bubble elastically recovers toward its relaxed position.  The residual
  x-displacement is the trapping/recovery metric.

Domain:
  x : 4·lref    periodic
  y : 2·lref     solid walls  (top = caprock stationary, bottom moves in shear)
  z : 2·lref     periodic

Physical parameters (H₂–brine at UHS conditions, ≈130 bar):
  ρ_H2      = 100  kg/m³   (density ratio 1:10 with brine)
  ρ_brine   = 1000 kg/m³
  μ_H2      = 1e-4 Pa·s
  μ_brine   = 1e-3 Pa·s

Key dimensionless numbers:
  Bond number  Bo = Δρ g R² / σ  ≈ 0.88   (buoyancy important)
  Capillary    Ca = μ_brine U_wall / σ = 0.001  (surface-tension dominated)
  Reynolds     Re = ρ_brine U_wall lref / μ_brine = 10

Usage:
    python3 run_h2_bubble_shear.py <num_length> <init_gsd_file> \\
            [theta_eq] [theta_adv] [theta_rec] \\
            [steps_relax] [steps_shear] [steps_snapback] [gsd_period] \\
            [ca] [sigma]

    num_length    : resolution used to create the init file (e.g. 20)
    init_gsd_file : path to h2brine_*_init.gsd (from create_input_geometry.py)
    theta_eq      : equilibrium contact angle [°]      (default 40)
    theta_adv     : advancing contact angle  [°]       (default 52)
    theta_rec     : receding  contact angle  [°]       (default 28)
    steps_relax   : relaxation steps                   (default 20001)
    steps_shear   : shear steps                        (default 50001)
    steps_snapback: snap-back steps                    (default 50001)
    gsd_period    : GSD write interval                 (default 200)
    ca            : capillary number (→ U_wall = ca·σ/μ_brine) (default 0.001)
    sigma         : surface tension [N/m]              (default 0.01)
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

num_length     = int(sys.argv[1])
filename       = str(sys.argv[2])
theta_eq       = float(sys.argv[3])  if len(sys.argv) > 3  else 40.0
theta_adv      = float(sys.argv[4])  if len(sys.argv) > 4  else 52.0
theta_rec      = float(sys.argv[5])  if len(sys.argv) > 5  else 28.0
steps_relax    = int(sys.argv[6])    if len(sys.argv) > 6  else 20001
steps_shear    = int(sys.argv[7])    if len(sys.argv) > 7  else 50001
steps_snapback = int(sys.argv[8])    if len(sys.argv) > 8  else 50001
gsd_period     = int(sys.argv[9])    if len(sys.argv) > 9  else 200
ca             = float(sys.argv[10]) if len(sys.argv) > 10 else 0.001
sigma          = float(sys.argv[11]) if len(sys.argv) > 11 else 0.01

dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logname   = filename.replace('_init.gsd', '_run.log')
dumpname  = filename.replace('_init.gsd', '_run.gsd')

sim.create_state_from_gsd(filename=filename)

# ─── Physical parameters ─────────────────────────────────────────────────────
lref       = 0.001        # reference length                               [m]
dx         = lref / num_length
rho01      = 100.0        # rest density phase W (H₂ bubble)              [kg/m³]
rho02      = 1000.0       # rest density phase N (brine)                  [kg/m³]
viscosity1 = 1e-4         # dynamic viscosity H₂                          [Pa·s]
viscosity2 = 1e-3         # dynamic viscosity brine                        [Pa·s]
gy         = -9.81        # gravitational acceleration (−y direction)      [m/s²]
backpress  = 0.01         # background pressure coefficient                [–]
drho       = 0.01         # allowed density variation                      [–]

# U_wall derived from capillary number: Ca = μ_brine · U_wall / σ
U_wall = ca * sigma / viscosity2

H_flu          = 2 * lref                                   # fluid channel height  [m]
R_drop_target  = H_flu / 2                                  # = lref                [m]
a_cube         = (2 * np.pi * R_drop_target**3 / 3) ** (1/3)  # ≈ 1.28·lref        [m]

# Derived dimensionless numbers (for information)
n_cube = int(round(a_cube / dx))
R_hemi = (3 * n_cube**3 * dx**3 / (2 * np.pi)) ** (1 / 3)
drho = rho02 - rho01   # density difference for buoyancy Bond number
Bo = drho * abs(gy) * R_hemi**2 / sigma
Ca = viscosity2 * U_wall / sigma       # should equal `ca`
Re = rho02 * U_wall * lref / viscosity2

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
filterfluidW = hoomd.filter.Type(['W'])   # H₂ bubble phase
filterfluidN = hoomd.filter.Type(['N'])   # brine phase
filtersolid  = hoomd.filter.Type(['S'])   # solid walls (bottom + caprock)

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
model.omega            = theta_eq     # equilibrium / nominal contact angle [°]
model.omega_adv        = theta_adv    # advancing contact angle             [°]
model.omega_rec        = theta_rec    # receding  contact angle             [°]
model.hysteresis       = True         # enable contact-angle hysteresis
model.gx               = 0.0
model.gy               = gy
model.damp             = 1000
model.artificialviscosity = True
model.alpha            = 0.2
model.beta             = 0.0
model.densitydiffusion = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c1, cond1, c2, cond2 = model.compute_speedofsound(
    LREF=lref, UREF=U_wall, DX=dx, DRHO=drho,
    H=maximum_smoothing_length,
    MU1=viscosity1, MU2=viscosity2,
    RHO01=rho01, RHO02=rho02, SIGMA12=sigma)

if device.communicator.rank == 0:
    print(f'Phase W (H₂)   speed of sound: {c1:.4f} m/s  ({cond1})')
    print(f'Phase N (brine) speed of sound: {c2:.4f} m/s  ({cond2})')

sph_helper.update_min_c0_tpf(device, model, c1, c2,
                              mode='plain', lref=lref, uref=U_wall, cfactor=10.0)

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=U_wall, DX=dx, DRHO=drho,
    H=maximum_smoothing_length,
    MU1=viscosity1, MU2=viscosity2,
    RHO01=rho01, RHO02=rho02, SIGMA12=sigma)

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')
    print(f'Bond number  Bo = {Bo:.4f}  (H₂ buoyancy vs surface tension)')
    print(f'Capillary    Ca = {Ca:.4f}  (U_wall = {U_wall:.4f} m/s)')
    print(f'Reynolds     Re = {Re:.1f}')
    print(f'θ_eq={theta_eq}°  θ_adv={theta_adv}°  θ_rec={theta_rec}°  Δθ={theta_adv-theta_rec:.1f}°')
    print(f'R_bubble (hemisphere) ≈ {R_hemi*1e3:.3f} mm')

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
# Remove any stale output GSD left by a previous crashed run.
# ALL ranks attempt the removal so each node's NFS metadata cache is flushed;
# the first rank to run succeeds, the rest get FileNotFoundError (ignored).
try:
    os.remove(dumpname)
except (FileNotFoundError, OSError):
    pass
device.communicator.barrier()

gsd_writer = hoomd.write.GSD(filename=dumpname,
                              trigger=hoomd.trigger.Periodic(gsd_period),
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
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(200), logger=logger,
                          max_header_len=10)
sim.operations.writers.append(table)

log_file = open(logname, mode='w+', newline='\n')
table_file = hoomd.write.Table(output=log_file,
                                trigger=hoomd.trigger.Periodic(200),
                                logger=logger, max_header_len=10)
sim.operations.writers.append(table_file)

# ─── Phase 1: Relaxation (stationary walls) ───────────────────────────────────
if device.communicator.rank == 0:
    print(f'\n── Phase 1: Relaxation  ({steps_relax} steps) ─────────────────────')
    print(f'   Both solid walls stationary.  H₂ cube morphs into spherical-cap')
    print(f'   bubble pressed against caprock (θ = {theta_eq}°, buoyancy active).')
    print(f'   Starting at {dt_string}')

sim.run(steps_relax, write_at_start=True)
gsd_writer.flush()

# ─── Measure bubble shape after relaxation ───────────────────────────────────
cx_r = cy_r = cz_r = 0.0
dy_r = dx_r = dz_r = 0.0
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap_r = traj[-1]

    pos_r = snap_r.particles.position
    tid_r = snap_r.particles.typeid
    bub_r = (tid_r == 0)   # 'W' — H₂ bubble

    if bub_r.any():
        p = pos_r[bub_r]
        cx_r = float(np.mean(p[:, 0]))
        cy_r = float(np.mean(p[:, 1]))
        cz_r = float(np.mean(p[:, 2]))
        dy_r = float(p[:, 1].max() - p[:, 1].min())   # bubble height (y-extent)
        dx_r = float(p[:, 0].max() - p[:, 0].min())   # bubble extent in x
        dz_r = float(p[:, 2].max() - p[:, 2].min())   # bubble extent in z
        print(f'\n── Relaxation complete at step {snap_r.configuration.step} ──')
        print(f'   Bubble centroid    : ({cx_r*1e3:.4f}, {cy_r*1e3:.4f}, {cz_r*1e3:.4f}) mm')
        print(f'   Bubble height (y)  : {dy_r*1e3:.4f} mm')
        print(f'   Bubble extent  x   : {dx_r*1e3:.4f} mm')
        print(f'   Bubble extent  z   : {dz_r*1e3:.4f} mm')
    else:
        print('   Warning: no H₂ bubble (W) particles found in final relaxation frame.')

# ─── Phase transition: activate bottom wall shear ────────────────────────────
# The BOTTOM wall moves; the TOP wall (caprock) remains stationary.
if device.communicator.rank == 0:
    print(f'\n── Phase transition: setting bottom wall velocity to {U_wall:.4f} m/s ──')

with sim.state.cpu_local_snapshot as snap:
    pos_l = np.asarray(snap.particles.position)
    tid_l = np.asarray(snap.particles.typeid)
    vel_l = np.asarray(snap.particles.velocity)
    # Bottom solid wall: type 'S' (index 2) and y < -H_flu/2
    bottom_mask = (tid_l == 2) & (pos_l[:, 1] < -H_flu / 2)
    vel_l[bottom_mask, 0] = U_wall   # set vx = U_wall; vy, vz remain zero

if device.communicator.rank == 0:
    print(f'   Bottom wall velocity updated to {U_wall:.4f} m/s.')

# ─── Phase 2: Shear (bottom wall moving at U_wall) ───────────────────────────
if device.communicator.rank == 0:
    print(f'\n── Phase 2: Shear  ({steps_shear} steps) ──────────────────────────')
    print(f'   Bottom wall velocity U_wall = {U_wall:.4f} m/s.')
    print(f'   Capillary number Ca = {Ca:.4f}.  Top caprock remains stationary.')

sim.run(steps_shear)
gsd_writer.flush()

# ─── Post-processing: bubble shape after shear ───────────────────────────────
cx_s = cy_s = cz_s = 0.0
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap_s = traj[-1]

    pos_s = snap_s.particles.position
    tid_s = snap_s.particles.typeid
    bub_s = (tid_s == 0)   # 'W' — H₂ bubble

    if bub_s.any():
        p = pos_s[bub_s]
        cx_s = float(np.mean(p[:, 0]))
        cy_s = float(np.mean(p[:, 1]))
        cz_s = float(np.mean(p[:, 2]))
        dy_s = float(p[:, 1].max() - p[:, 1].min())
        dx_s = float(p[:, 0].max() - p[:, 0].min())
        dz_s = float(p[:, 2].max() - p[:, 2].min())
        # Deformation parameter D = (L - B) / (L + B)
        L = max(dx_s, dy_s)
        B = min(dx_s, dy_s)
        D_param = (L - B) / (L + B) if (L + B) > 0 else 0.0
        delta_x_shear = cx_s - cx_r
        # Spherical-cap contact angle estimate (bubble on ceiling, extends downward)
        # R = (r² + h²)/(2h),  cos(θ_cap) = (R - h)/R = 1 - 2h²/(r² + h²)
        r_cap = dx_s / 2.0
        h_cap = dy_s
        if r_cap**2 + h_cap**2 > 0:
            theta_cap = float(np.degrees(np.arccos(
                np.clip(1.0 - 2.0*h_cap**2 / (r_cap**2 + h_cap**2), -1, 1))))
        else:
            theta_cap = float('nan')
        print(f'\n── Shear complete at step {snap_s.configuration.step} ──')
        print(f'   Bubble centroid    : ({cx_s*1e3:.4f}, {cy_s*1e3:.4f}, {cz_s*1e3:.4f}) mm')
        print(f'   Bubble height (y)  : {dy_s*1e3:.4f} mm')
        print(f'   Bubble extent  x   : {dx_s*1e3:.4f} mm')
        print(f'   Bubble extent  z   : {dz_s*1e3:.4f} mm')
        print(f'   Deformation D      : {D_param:.4f}  (0 = undeformed, 1 = fully elongated)')
        print(f'   Centroid x-drift   : Δx = {delta_x_shear*1e3:.4f} mm')
        print(f'   Spherical-cap θ    : {theta_cap:.2f}°  (estimated from extent)')
    else:
        print('   Warning: no H₂ bubble (W) particles found in final shear frame.')

# ─── Phase transition: deactivate bottom wall (snap-back) ────────────────────
if device.communicator.rank == 0:
    print(f'\n── Phase transition: resetting bottom wall velocity to 0 m/s ──')

with sim.state.cpu_local_snapshot as snap:
    pos_l = np.asarray(snap.particles.position)
    tid_l = np.asarray(snap.particles.typeid)
    vel_l = np.asarray(snap.particles.velocity)
    bottom_mask = (tid_l == 2) & (pos_l[:, 1] < -H_flu / 2)
    vel_l[bottom_mask, 0] = 0.0

if device.communicator.rank == 0:
    print(f'   Bottom wall velocity reset to zero.')

# ─── Phase 3: Snap-back (bottom wall stationary again) ───────────────────────
if device.communicator.rank == 0:
    print(f'\n── Phase 3: Snap-back  ({steps_snapback} steps) ────────────────────')
    print(f'   Bottom wall stopped.  Hysteresis pinning allows elastic recovery.')
    print(f'   Residual x-drift < 20% of shear drift → "full recovery" verdict.')

sim.run(steps_snapback)
gsd_writer.flush()

# ─── Post-processing: bubble shape after snap-back ────────────────────────────
if device.communicator.rank == 0:
    with gsd.hoomd.open(dumpname, 'r') as traj:
        snap_b = traj[-1]

    pos_b = snap_b.particles.position
    tid_b = snap_b.particles.typeid
    bub_b = (tid_b == 0)   # 'W' — H₂ bubble

    if bub_b.any():
        p = pos_b[bub_b]
        cx_b = float(np.mean(p[:, 0]))
        cy_b = float(np.mean(p[:, 1]))
        cz_b = float(np.mean(p[:, 2]))
        dy_b = float(p[:, 1].max() - p[:, 1].min())
        dx_b = float(p[:, 0].max() - p[:, 0].min())
        dz_b = float(p[:, 2].max() - p[:, 2].min())
        L = max(dx_b, dy_b)
        B = min(dx_b, dy_b)
        D_param_b = (L - B) / (L + B) if (L + B) > 0 else 0.0
        delta_x_shear    = cx_s - cx_r     # drift during shear
        delta_x_recovery = cx_b - cx_r     # residual after snap-back
        print(f'\n── Snap-back complete at step {snap_b.configuration.step} ──')
        print(f'   Bubble centroid    : ({cx_b*1e3:.4f}, {cy_b*1e3:.4f}, {cz_b*1e3:.4f}) mm')
        print(f'   Bubble height (y)  : {dy_b*1e3:.4f} mm')
        print(f'   Bubble extent  x   : {dx_b*1e3:.4f} mm')
        print(f'   Bubble extent  z   : {dz_b*1e3:.4f} mm')
        print(f'   Deformation D      : {D_param_b:.4f}  (0 = undeformed, 1 = fully elongated)')
        print(f'   Centroid x-drift (shear)    : Δx_shear    = {delta_x_shear*1e3:.4f} mm')
        print(f'   Centroid x-drift (residual) : Δx_recovery = {delta_x_recovery*1e3:.4f} mm')
        if abs(delta_x_shear) > 0 and abs(delta_x_recovery) < abs(delta_x_shear) * 0.2:
            verdict = "Full recovery (hysteresis pinning effective — bubble stable at caprock)"
        else:
            verdict = "Partial recovery (residual displacement — potential trapping shift)"
        print(f'   Verdict            : {verdict}')
    else:
        print('   Warning: no H₂ bubble (W) particles found in final snap-back frame.')

    print(f'\nOutput GSD : {dumpname}')
    print(f'Output log : {logname}')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_tpf(dumpname)
