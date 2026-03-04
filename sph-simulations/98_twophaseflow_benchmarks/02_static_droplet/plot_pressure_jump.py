#!/usr/bin/env python3
"""
Plot pressure-jump diagnostics for the static-droplet benchmark.

Two panels are produced and saved as pressure_jump.png:

  Left  – Radial pressure profile (last frame)
            Scatter of per-particle pressure vs. radial distance from the
            droplet centre.  Phase A (outer) and phase B (droplet) are
            coloured separately.  Horizontal lines mark the mean pressures;
            a dashed vertical line marks the nominal droplet radius.

  Right – Pressure-jump convergence
            Mean ΔP = P_in − P_out sampled every <stride> VTU frames,
            compared to the Young–Laplace theory value  ΔP = 2σ/R.

Usage:
    python3 plot_pressure_jump.py [run_dir] [stride]

    run_dir  directory containing the .vtu files  (default: auto-detect)
    stride   read every N-th vtu file             (default: 10)
"""

import sys
import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vtk

# ── Physical parameters (must match run script) ───────────────────────────────
LREF    = 0.001          # reference length  [m]
R_DROP  = 0.25 * LREF   # droplet radius    [m]
SIGMA   = 0.01           # surface tension   [N/m]
MU      = 0.001          # dynamic viscosity [Pa·s]

dP_theory = 2.0 * SIGMA / R_DROP   # 80 Pa
U_cap     = SIGMA / MU              # capillary velocity scale [m/s]

# ── Locate VTU directory ──────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:
    run_dir = sys.argv[1]
else:
    # auto-detect: look for a _run directory next to this script
    candidates = sorted(glob.glob(os.path.join(script_dir, '*_run')))
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        raise FileNotFoundError(
            'No *_run directory found. Pass the directory as the first argument.')
    run_dir = candidates[0]

stride = int(sys.argv[2]) if len(sys.argv) > 2 else 10

vtu_files = sorted(glob.glob(os.path.join(run_dir, '*.vtu')))
if not vtu_files:
    raise FileNotFoundError(f'No .vtu files found in {run_dir}')

print(f'Run directory : {run_dir}')
print(f'VTU files     : {len(vtu_files)}  (stride={stride})')
print(f'ΔP theory     : {dP_theory:.1f} Pa')

# ── Helper: read one VTU frame ────────────────────────────────────────────────
def read_vtu(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    ug  = reader.GetOutput()
    pts = np.array([ug.GetPoint(i) for i in range(ug.GetNumberOfPoints())])
    pd  = ug.GetPointData()
    tid  = np.array(pd.GetArray('TypeId')).ravel()
    pres = np.array(pd.GetArray('Pressure')).ravel()
    return pts, tid, pres

def extract_step(path):
    m = re.search(r'_(\d+)\.vtu$', path)
    return int(m.group(1)) if m else -1

# ── 1. Radial pressure profile from last frame ────────────────────────────────
print('Reading last frame …')
pts_last, tid_last, pres_last = read_vtu(vtu_files[-1])
step_last = extract_step(vtu_files[-1])

# droplet centre — centroid of phase-B particles
centre = pts_last[tid_last == 1].mean(axis=0)
r_all  = np.linalg.norm(pts_last - centre, axis=1)

# thin the scatter for readability (max 8 000 points per phase)
rng = np.random.default_rng(42)
def thin(mask, n=8000):
    idx = np.where(mask)[0]
    if len(idx) > n:
        idx = rng.choice(idx, n, replace=False)
    return idx

idx_A = thin(tid_last == 0)
idx_B = thin(tid_last == 1)

inside  = tid_last == 1
P_in    = float(np.mean(pres_last[inside]))
P_out   = float(np.mean(pres_last[~inside]))
dP_sim  = P_in - P_out
rel_err = abs(dP_sim - dP_theory) / dP_theory * 100.0

# ── 2. ΔP convergence over all sampled frames ─────────────────────────────────
sampled_files = vtu_files[::stride]
steps_ts  = []
dP_ts     = []
P_in_ts   = []
P_out_ts  = []

print(f'Sampling {len(sampled_files)} frames for time-series …')
for i, fpath in enumerate(sampled_files):
    _, tid_f, pres_f = read_vtu(fpath)
    mask_in = tid_f == 1
    Pi  = float(np.mean(pres_f[mask_in]))
    Po  = float(np.mean(pres_f[~mask_in]))
    steps_ts.append(extract_step(fpath))
    P_in_ts.append(Pi)
    P_out_ts.append(Po)
    dP_ts.append(Pi - Po)
    if (i + 1) % 20 == 0 or i == len(sampled_files) - 1:
        print(f'  [{i+1}/{len(sampled_files)}]  step {steps_ts[-1]:>8d}  '
              f'ΔP = {dP_ts[-1]:.2f} Pa')

steps_ts  = np.array(steps_ts)
dP_ts     = np.array(dP_ts)
P_in_ts   = np.array(P_in_ts)
P_out_ts  = np.array(P_out_ts)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Static-droplet benchmark — pressure jump', fontsize=13, y=1.01)

# --- Left: radial pressure profile ---
ax = axes[0]
ax.scatter(r_all[idx_A] * 1e3, pres_last[idx_A],
           s=3, alpha=0.4, color='steelblue', label='Phase A (outer)', rasterized=True)
ax.scatter(r_all[idx_B] * 1e3, pres_last[idx_B],
           s=3, alpha=0.6, color='tomato', label='Phase B (droplet)', rasterized=True)

ax.axhline(P_in,  color='tomato',    lw=1.5, ls='--',
           label=f'$\\bar{{P}}_{{in}}$  = {P_in:.1f} Pa')
ax.axhline(P_out, color='steelblue', lw=1.5, ls='--',
           label=f'$\\bar{{P}}_{{out}}$ = {P_out:.1f} Pa')
ax.axvline(R_DROP * 1e3, color='k', lw=1.0, ls=':', label=f'$R$ = {R_DROP*1e3:.2f} mm')

ax.set_xlabel('Radial distance $r$  [mm]')
ax.set_ylabel('Pressure  [Pa]')
ax.set_title(f'Radial profile  (step {step_last:,})\n'
             f'$\\Delta P_{{sim}}$ = {dP_sim:.1f} Pa,  '
             f'$\\Delta P_{{theory}}$ = {dP_theory:.1f} Pa  '
             f'(err {rel_err:.1f} %)')
ax.legend(fontsize=8, markerscale=4)
ax.grid(True, alpha=0.3)

# --- Right: ΔP vs timestep ---
ax = axes[1]
ax.plot(steps_ts, dP_ts, color='royalblue', lw=1.5, label='$\\Delta P$ (simulation)')
ax.axhline(dP_theory, color='k', lw=1.2, ls='--',
           label=f'$\\Delta P_{{theory}}$ = {dP_theory:.1f} Pa  ($2\\sigma/R$)')

# shade the band between P_in and P_out to show absolute levels
ax.fill_between(steps_ts, P_in_ts, P_out_ts, alpha=0.12, color='royalblue')

ax.set_xlabel('Timestep')
ax.set_ylabel('$\\Delta P = P_{in} - P_{out}$  [Pa]')
ax.set_title('Pressure-jump convergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = os.path.join(script_dir, 'pressure_jump.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'\nSaved → {outpath}')
