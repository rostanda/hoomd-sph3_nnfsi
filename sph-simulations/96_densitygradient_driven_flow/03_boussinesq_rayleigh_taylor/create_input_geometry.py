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

Create initial GSD file for the Boussinesq Rayleigh–Taylor instability benchmark.

GEOMETRY
--------
Tall cavity of width $lref \times height\ 4 \cdot lref$ enclosed by solid walls at top and bottom.
  - Fluid (type 'F', typeid 0): |y| < 2*lref
  - Solid (type 'S', typeid 1): |y| >= 2*lref  (wall layers, n_solid thick)

The domain is periodic in x (one perturbation wavelength) and z (thin quasi-2D slice).
Temperature assignment (T stored in aux4.x) is done in the run script:
  - Upper fluid ($y > \delta \cdot \cos(2\pi \cdot x/lref)$): T = 0   (cold, heavy → sinks)
  - Lower fluid ($y \leq \delta \cdot \cos(2\pi \cdot x/lref)$): T = 1   (hot,  light → rises)
  - $\delta = 0.1 \cdot lref$ is the initial perturbation amplitude.

Physical parameters (set in run script):
    lref      = 0.001 m    perturbation wavelength = cavity width
    rho0      = 1000 kg/m$^3$ rest density
    beta_s    = 0.5        Boussinesq thermal expansion coefficient
    gy        = -9.81 m/s$^2$ gravity
    At_eff    = $\beta_s \cdot \Delta T / 2$ = 0.25  (effective Atwood number)

Linear RT growth rate:  $\gamma = \sqrt{At\_eff \cdot |gy| \cdot 2\pi/lref}$

Usage:
    python3 create_input_geometry.py <num_length>

    num_length : fluid particles across the cavity width lref (e.g. 20)
"""

import sys, math
import numpy as np
import gsd.hoomd

import hoomd
from hoomd import sph

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = int(sys.argv[1])

lref   = 0.001       # perturbation wavelength = cavity width  [m]
rho0   = 1000.0      # rest density                            [kg/m³]
dx     = lref / num_length
mass   = rho0 * dx**3

kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength

n_solid    = math.ceil(rcut / dx)              # wall thickness [particle layers]
part_depth = math.ceil(2.5 * hoomd.sph.kernel.Kappa[kernel] * rcut / dx)  # $\ell_z > 2\kappa h$ (pseudo-3D: more than one kernel diameter in z)

# Domain: lref wide (periodic in x), 4*lref tall + walls (solid in y), thin (periodic in z)
nx = num_length
ny = 4 * num_length + 2 * n_solid
nz = part_depth

lx = nx * dx
ly = ny * dx
lz = nz * dx

# ─── Particle positions (half-integer offsets — no particles on box boundaries) ──
xs = np.linspace(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
ys = np.linspace(-ly / 2 + dx / 2, ly / 2 - dx / 2, ny)
zs = np.linspace(-lz / 2 + dx / 2, lz / 2 - dx / 2, nz)

xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
positions   = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
n_particles = positions.shape[0]

velocities = np.zeros((n_particles, 3), dtype=np.float32)
masses     = np.full(n_particles, mass,    dtype=np.float32)
slengths   = np.full(n_particles, slength, dtype=np.float32)
densities  = np.full(n_particles, rho0,    dtype=np.float32)

# ─── Type assignment: fluid inside |y| < 2*lref, solid outside ───────────────
# With half-integer offsets, no particle lands exactly at $\pm 2 \cdot l_\mathrm{ref}$.
# Bottom solid: y[k] = -ly/2 + (k+0.5)*dx  →  y[n_solid-1] = -2*lref - dx/2  (< -2*lref)
# Bottom fluid: y[n_solid]                  = -2*lref + dx/2                  (> -2*lref)
y_pos    = positions[:, 1]
is_solid = (y_pos < -2.0 * lref) | (y_pos > 2.0 * lref)
typeid   = np.where(is_solid, 1, 0).astype(np.int32)

# ─── Write GSD ───────────────────────────────────────────────────────────────
snapshot = gsd.hoomd.Frame()
snapshot.configuration.box  = [lx, ly, lz, 0, 0, 0]
snapshot.particles.N        = n_particles
snapshot.particles.types    = ['F', 'S']
snapshot.particles.typeid   = typeid
snapshot.particles.position = positions.astype(np.float32)
snapshot.particles.velocity = velocities
snapshot.particles.mass     = masses
snapshot.particles.slength  = slengths
snapshot.particles.density  = densities

init_filename = f'boussinesq_rt_{num_length}_dx_{dx:.2e}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

n_fluid   = int(np.sum(typeid == 0))
n_solid_p = int(np.sum(typeid == 1))

# ─── Print summary with linear-theory reference values ───────────────────────
At_eff    = 0.25              # beta_s * DeltaT / 2 = 0.5 * 1.0 / 2
k_wave    = 2.0 * np.pi / lref
gamma_lin = np.sqrt(At_eff * 9.81 * k_wave)
delta     = 0.1 * lref

print(f'Written {init_filename}: {n_fluid} fluid, {n_solid_p} solid ({n_particles} total)')
print(f'  Domain:  {lx*1e3:.2f} × {ly*1e3:.2f} × {lz*1e3:.3f} mm  (nx={nx}, ny={ny}, nz={nz})')
print(f'  dx = {dx*1e6:.1f} µm,  slength = {slength*1e6:.1f} µm,  rcut = {rcut*1e6:.1f} µm')
print(f'  Fluid height = {4*lref*1e3:.1f} mm  (±{2*lref*1e3:.1f} mm),  wall layers = {n_solid}')
print(f'  Linear RT:  At_eff = {At_eff:.3f},  γ = {gamma_lin:.1f} s⁻¹,  τ_lin = {1/gamma_lin*1e3:.2f} ms')
print(f'  Perturbation δ = {delta*1e6:.1f} µm  ({delta/dx:.1f} particle spacings)')
print(f'  Temperature assigned in run script via cpu_local_snapshot.')
