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

BCC sphere packing — permeability & inertia-onset run script.

BENCHMARK DESCRIPTION
---------------------
Body-force-driven flow through a periodic BCC sphere packing
(100$\times$100$\times$100 voxels, 4$\times$4$\times$4 unit cells, a = 25 vox, R = 10 vox, $\varphi \approx 0.464$).

By sweeping the body force fx over many orders of magnitude (see run_all_re.sh),
the full Darcy–Forchheimer curve is sampled:

  Darcy regime      ($Re_\mathrm{grain} \ll 1$)   : k = const = $k_\mathrm{Darcy}$
  Onset of inertia  ($Re_\mathrm{grain} \approx 1$–10) : k begins to decrease
  Forchheimer regime ($Re_\mathrm{grain} > 10$)  : k decreases as $1/(1 + \beta k\, Re_\mathrm{grain} / \mu)$

KEY QUANTITIES LOGGED
---------------------
  $Re_\mathrm{grain} = \rho \varphi \langle u \rangle d / \mu$       (grain-scale Reynolds number)
               where $\langle u \rangle$ = abs_velocity (mean fluid speed) and d = 2R
  $k        = \mu \varphi \langle u \rangle / (\rho f_x)$    (apparent Darcy permeability, m$^2$)
  $k_\mathrm{norm}   = k / k_\mathrm{KC}$             (normalised by Kozeny–Carman estimate)
               $k_\mathrm{norm} \approx 1$ in Darcy regime; $< 1$ in Forchheimer regime

SPEED OF SOUND / CFL
---------------------
The reference velocity for the CFL estimate is taken from the Kozeny–Carman
Darcy prediction: $U_\mathrm{ref} = k_\mathrm{KC} \rho f_x / \mu$.  In the Forchheimer regime this
overestimates the actual velocity (actual Re is lower than the linear Darcy
prediction), so the resulting c_s and dt are conservative.

Note on `damp`:
  model.damp is a body-force ramp time (in time steps).  The body force is
  smoothly ramped from 0 to fx over the first `damp` steps using a sinusoidal
  envelope.  It does NOT add velocity-dependent drag and does NOT affect the
  steady-state permeability measurement.

Usage:
    python3 run_bcc_permeability.py <init_gsd> <fx> [steps] [damp]
      init_gsd : path to bcc100_init.gsd
      fx       : body force in x-direction [m/s$^2$]  (positive $\to$ +x)
      steps    : simulation steps  (default: 50001)
      damp     : body-force ramp-up time in steps  (default: 5000)
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

filename = str(sys.argv[1])
fx       = float(sys.argv[2])
steps    = int(sys.argv[3])   if len(sys.argv) > 3 else 50001
damp     = int(sys.argv[4])   if len(sys.argv) > 4 else 5000

dt_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
tag       = f'fx{fx:.4e}'
logname   = filename.replace('_init.gsd', f'_{tag}_run.log')
dumpname  = filename.replace('_init.gsd', f'_{tag}_run.gsd')

sim.create_state_from_gsd(filename=filename)

# ─── Physical parameters ─────────────────────────────────────────────────────
# Geometry constants matching create_bcc_geometry.py
NX      = 100
A_VOX   = 25           # unit cell lattice constant [voxels]
R_VOX   = 10           # sphere radius [voxels]

rho0      = 1000.0     # rest density              [kg/m³]
viscosity = 0.001      # dynamic viscosity         [Pa·s]
drho      = 0.01       # allowed density variation [–]
backpress = 0.01       # background pressure coeff [–]

# Count fluid particles to get porosity from the actual geometry.
# get_snapshot() gathers all particles on rank 0; other ranks see N=0.
phi_val      = 0.0
vsize_val    = 0.0
n_fluid_val  = 0.0
n_total_val  = 0.0
snapshot = sim.state.get_snapshot()
if snapshot.communicator.rank == 0:
    n_total_val  = float(len(snapshot.particles.typeid))
    n_fluid_val  = float(np.sum(snapshot.particles.typeid == 0))
    phi_val      = n_fluid_val / n_total_val
    vsize_val    = float(np.max(snapshot.particles.slength)) \
                   / hoomd.sph.kernel.OptimalH['WendlandC4']

phi     = device.communicator.bcast_double(phi_val)
vsize   = device.communicator.bcast_double(vsize_val)
n_total = int(round(device.communicator.bcast_double(n_total_val)))
n_fluid = int(round(device.communicator.bcast_double(n_fluid_val)))

lref    = NX * vsize           # domain length in x (= reference length)  [m]
dx      = vsize                # voxel / particle spacing                  [m]
d_grain = 2 * R_VOX * vsize   # grain diameter                            [m]

# Kozeny–Carman permeability estimate (Darcy-regime reference)
k_KC = d_grain**2 * phi**3 / (180.0 * (1.0 - phi)**2)

# Ergun inertia coefficient (Forchheimer $\beta$): $\beta = 1.75(1-\phi)/(d\phi^3)$
beta_ergun = 1.75 * (1.0 - phi) / (d_grain * phi**3)

# Reference velocity for CFL / c_s estimation.
# In the Darcy regime use the KC Darcy velocity.  In the Forchheimer regime the
# Darcy estimate can overestimate the actual velocity by orders of magnitude,
# giving an unnecessarily small dt.  Cap with the inertia-limited velocity
#   $U_\mathrm{D,inertia} = \sqrt{\rho f_x / (\beta \rho)} = \sqrt{f_x / \beta}$
# which is the superficial velocity when the quadratic Forchheimer term
# balances the body force.  The smaller of the two estimates is more accurate.
U_D_KC      = k_KC * rho0 * fx / viscosity          # Darcy prediction [m/s]
U_D_inertia = np.sqrt(fx / beta_ergun)              # inertia limit    [m/s]
U_D_ref     = min(U_D_KC, U_D_inertia)              # tighter estimate
refvel      = max(U_D_ref / phi, 1e-10)             # mean pore velocity [m/s]

if device.communicator.rank == 0:
    Re_KC     = rho0 * U_D_KC * d_grain / viscosity
    Re_inertia = rho0 * U_D_inertia * d_grain / viscosity
    print(f'BCC permeability run:  fx = {fx:.4e} m/s²')
    print(f'  vsize = {vsize:.4e} m,  d_grain = {d_grain:.4e} m,  lref = {lref:.4e} m')
    print(f'  φ = {phi:.4f},  n_fluid = {n_fluid},  n_total = {n_total}')
    print(f'  k_KC = {k_KC:.4e} m²  ({k_KC/9.869e-13:.2f} Darcy)')
    print(f'  β_Ergun = {beta_ergun:.2f} m⁻¹')
    print(f'  U_D_KC = {U_D_KC:.4e} m/s  (Re_grain = {Re_KC:.4e})  [Darcy prediction]')
    print(f'  U_D_inertia = {U_D_inertia:.4e} m/s  (Re_grain = {Re_inertia:.4e})  [inertia limit]')
    print(f'  refvel used = {refvel:.4e} m/s  (min of both)')

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
model = hoomd.sph.sphmodel.SinglePhaseFlow(
    kernel=kernel_obj, eos=eos, nlist=nlist,
    fluidgroup_filter=filterfluid, solidgroup_filter=filtersolid,
    densitymethod='SUMMATION')

model.mu                    = viscosity
model.gx                    = fx
model.damp                  = damp        # body-force ramp-up time [steps]
model.artificialviscosity   = True
model.alpha                 = 0.2
model.beta                  = 0.2
model.densitydiffusion      = False
model.shepardrenormanlization = False

# ─── Speed of sound & timestep ───────────────────────────────────────────────
maximum_smoothing_length = sph_helper.set_max_sl(sim, device, model)

c, cond = model.compute_speedofsound(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

if device.communicator.rank == 0:
    print(f'Speed of sound: {c:.4f} m/s  ({cond})')

dt, dt_cond = model.compute_dt(
    LREF=lref, UREF=refvel, DX=dx, DRHO=drho,
    H=maximum_smoothing_length, MU=viscosity, RHO0=rho0)

if device.communicator.rank == 0:
    print(f'Timestep: {dt:.3e} s  ({dt_cond})')
    print(f'Total simulated time: {steps * dt:.3e} s  ({steps} steps)')

# ─── Integrator ──────────────────────────────────────────────────────────────
integrator = hoomd.sph.Integrator(dt=dt)
vvb = hoomd.sph.methods.VelocityVerletBasic(filter=filterfluid,
                                              densitymethod='SUMMATION')
integrator.methods.append(vvb)
integrator.forces.append(model)
sim.operations.integrator = integrator

# ─── Computes ────────────────────────────────────────────────────────────────
spf_properties = hoomd.sph.compute.SinglePhaseFlowBasicProperties(filterfluid)
sim.operations.computes.append(spf_properties)

# ─── Output ──────────────────────────────────────────────────────────────────
# Remove any stale output GSD left by a previous crashed run.
# ALL ranks attempt the removal so each node's NFS metadata cache is flushed;
# the first rank to run succeeds, the rest get FileNotFoundError (ignored).
try:
    os.remove(dumpname)
except (FileNotFoundError, OSError):
    pass
device.communicator.barrier()

gsd_period = max(1, steps // 100)   # ≈ 100 frames total
gsd_writer = hoomd.write.GSD(filename=dumpname,
                              trigger=hoomd.trigger.Periodic(gsd_period),
                              mode='wb',
                              dynamic=['property', 'momentum'])
sim.operations.writers.append(gsd_writer)

logger = hoomd.logging.Logger(categories=['scalar', 'string'])
logger.add(sim, quantities=['timestep', 'tps', 'walltime'])
logger.add(spf_properties,
           quantities=['abs_velocity', 'num_particles', 'fluid_vel_x_sum',
                       'mean_density', 'e_kin_fluid'])

# Grain-scale Reynolds number: $Re = \rho \phi \langle u \rangle d / \mu$
logger[('custom', 'Re_grain')] = (
    lambda: rho0 * phi * spf_properties.abs_velocity * d_grain / viscosity,
    'scalar')

# Apparent Darcy permeability: $k = \mu \phi \langle u \rangle / (\rho f_x)$  [$\mathrm{m}^2$]
logger[('custom', 'k_m2')] = (
    lambda: viscosity * phi * spf_properties.abs_velocity / (rho0 * fx),
    'scalar')

# Normalised permeability: k / k_KC  (1.0 in Darcy regime, < 1.0 in Forchheimer)
logger[('custom', 'k_norm')] = (
    lambda: viscosity * phi * spf_properties.abs_velocity / (rho0 * fx * k_KC),
    'scalar')

log_period = max(1, steps // 500)   # ≈ 500 log entries total
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(log_period),
                          logger=logger, max_header_len=10)
sim.operations.writers.append(table)

log_file = open(logname, mode='w+', newline='\n')
table_file = hoomd.write.Table(output=log_file,
                                trigger=hoomd.trigger.Periodic(log_period),
                                logger=logger, max_header_len=10)
sim.operations.writers.append(table_file)

# ─── Run ─────────────────────────────────────────────────────────────────────
if device.communicator.rank == 0:
    print(f'Starting BCC permeability run  fx={fx:.4e}  at {dt_string}')
sim.run(steps, write_at_start=True)
gsd_writer.flush()

# ─── Post-processing: steady-state Re and k ──────────────────────────────────
if device.communicator.rank == 0:
    # Read the last 25 % of log entries (after body force is fully applied)
    # to estimate the steady-state Re_grain and k
    import csv, io
    log_file.seek(0)
    raw = log_file.read()
    log_file.close()

    lines = [l for l in raw.splitlines() if l and not l.startswith('#')]
    # Find header line (contains 'Re_grain')
    header_idx = next((i for i, l in enumerate(lines) if 'Re_grain' in l), None)

    if header_idx is not None:
        header  = lines[header_idx].split()
        data_lines = [l for l in lines[header_idx + 1:] if l[0].isdigit() or l[0] == '-']
        n_tail = max(1, len(data_lines) // 4)   # last 25 %
        tail   = data_lines[-n_tail:]

        col_re = header.index('Re_grain')
        col_k  = header.index('k_m2')
        col_kn = header.index('k_norm')

        re_vals = [float(row.split()[col_re]) for row in tail]
        k_vals  = [float(row.split()[col_k])  for row in tail]
        kn_vals = [float(row.split()[col_kn]) for row in tail]

        re_ss = float(np.mean(re_vals))
        k_ss  = float(np.mean(k_vals))
        kn_ss = float(np.mean(kn_vals))

        # Forchheimer coefficient estimate: $\beta = (k_\mathrm{KC}/k - 1) / (\rho U_D d / \mu)$
        U_D_ss = k_ss * rho0 * fx / viscosity   # check: $U_D = k \rho f / \mu$
        beta_F = (k_KC / k_ss - 1.0) * viscosity / (rho0 * U_D_ss * d_grain) \
                  if re_ss > 0.1 else float('nan')

        print(f'\n── BCC permeability summary  (fx = {fx:.4e} m/s²) ──')
        print(f'  Steady-state Re_grain     = {re_ss:.4e}')
        print(f'  Steady-state k            = {k_ss:.4e} m²  '
              f'({k_ss/9.869e-13:.4f} Darcy)')
        print(f'  Steady-state k/k_KC       = {kn_ss:.4f}')
        print(f'  Kozeny–Carman k_KC        = {k_KC:.4e} m²')
        if not np.isnan(beta_F):
            print(f'  Forchheimer β estimate    = {beta_F:.4f} m⁻¹')
        print(f'  (Darcy-regime prediction  Re = {rho0*U_D_KC*d_grain/viscosity:.4e})')

        # Append to summary file for batch post-processing
        summary_file = filename.replace('_init.gsd', '_permeability_summary.dat')
        write_header = not os.path.exists(summary_file)
        with open(summary_file, 'a') as sf:
            if write_header:
                sf.write('# fx[m/s2]  Re_grain  k[m2]  k_norm  k_KC[m2]  phi\n')
            sf.write(f'{fx:.6e}  {re_ss:.6e}  {k_ss:.6e}  {kn_ss:.6e}  '
                     f'{k_KC:.6e}  {phi:.6f}\n')
    else:
        log_file.close()
        print('Warning: could not parse log file for post-processing.')

if HAS_VTU and device.communicator.rank == 0:
    export_gsd2vtu.export_spf(dumpname)
