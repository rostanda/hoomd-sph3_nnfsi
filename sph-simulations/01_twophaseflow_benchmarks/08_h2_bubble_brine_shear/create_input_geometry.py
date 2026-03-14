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

H$_2$ Bubble in Brine Under Shear — initial geometry creation.

BENCHMARK DESCRIPTION
---------------------
A cube of fluid 'W' (H$_2$ bubble phase) is placed in contact with the TOP solid
wall (caprock), immersed in the ambient fluid 'N' (brine).  Surface tension ($\sigma$)
and the prescribed contact angle ($\theta = 40°$, hydrophobic caprock) morph the cube
into a sessile spherical-cap bubble during the relaxation phase.  H$_2$ is
buoyant ($\rho_{H2}$ = 100 kg/m$^3$ $\ll$ $\rho_{brine}$ = 1000 kg/m$^3$) so it presses against the
caprock under gravity.

After equilibration the bottom solid wall is moved at velocity U_wall in the
x direction (Couette-like driving) to drive flow past the bubble.  Contact-angle
hysteresis pins the contact line.  Three phases: relax → shear → snap-back.

Underground Hydrogen Storage (UHS) context:
  Hydrogen at ~130 bar subsurface conditions has $\rho_{H2} \approx$ 100 kg/m$^3$.
  The density ratio $\rho_{H2} / \rho_{brine} \approx$ 1:10 is characteristic of UHS pore-scale.

Domain:
  x : $4 \cdot lref$    periodic  (room for elongation under shear)
  y : $2 \cdot lref$     solid walls  (top = caprock stationary, bottom moves in shear)
  z : $2 \cdot lref$     periodic

H$_2$ bubble cube:
  side length  $a = (2\pi/3)^{1/3} \cdot lref \approx 1.28 \cdot lref$  (volume-conserving for hemisphere)
  centred at   x = 0, z = 0
  top face     at $y = +H\_flu/2$  (in contact with the top caprock wall)

Solid walls:
  n_solid = 3 layers at bottom (moving in shear phase) and top (stationary caprock)
  All solid velocities are zero in the init file.
  The bottom wall velocity is set at runtime before the shear phase.

Key dimensionless numbers (default parameters):
  Bond number  $Bo = \Delta\rho\, g R^2 / \sigma \approx 0.88$   (buoyancy presses bubble on caprock)
  Capillary    $Ca = \mu_{brine} U\_wall / \sigma = 0.001$  (surface-tension dominated)
  Reynolds     $Re = \rho_{brine} U\_wall\, lref / \mu_{brine} = 10$

Usage:
    python3 create_input_geometry.py <num_length>

    num_length : integer resolution — particles across lref (e.g. 20, 40)

Output:
    h2brine_<nx>_<ny>_<nz>_vs_<dx>_init.gsd
"""

import sys
import numpy as np
import gsd.hoomd
import hoomd
from hoomd import sph

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = int(sys.argv[1]) if len(sys.argv) > 1 else 20

lref    = 0.001                    # reference length                      [m]
rho01   = 100.0                    # rest density phase W (H₂ bubble)      [kg/m³]
rho02   = 1000.0                   # rest density phase N (brine)           [kg/m³]
dx      = lref / num_length        # particle spacing                       [m]
mass1   = rho01 * dx**3            # particle mass phase W (H₂)             [kg]
mass2   = rho02 * dx**3            # particle mass phase N (brine)          [kg]
n_solid = 3                        # solid wall layers per side

kernel  = 'WendlandC4'
slength = hoomd.sph.kernel.OptimalH[kernel] * dx   # smoothing length      [m]

# ─── Box geometry ────────────────────────────────────────────────────────────
# Fluid region : $4 \cdot l_\mathrm{ref}$ (x) $\times$ $2 \cdot l_\mathrm{ref}$ (y) $\times$ $2 \cdot l_\mathrm{ref}$ (z)
# Solid layers : n_solid rows at bottom and top in y.
# x, z         : periodic.   y : bounded by solid walls.
nx     = 4 * num_length            # wider in x for shear-driven elongation
ny_flu = 2 * num_length            # fluid rows in y  (channel height = 2·lref)
ny     = ny_flu + 2 * n_solid      # total rows including solid walls
nz     = 2 * num_length

lx = nx * dx                       # 4·lref
ly = ny * dx
lz = nz * dx                       # 2·lref

H_flu = ny_flu * dx                # fluid channel height = 2·lref          [m]

# Bubble cube: volume-conserving for a hemisphere of radius R = H_flu/2
R_drop_target = H_flu / 2                              # = lref              [m]
a_cube = (2 * np.pi * R_drop_target**3 / 3) ** (1/3)  # ≈ 1.28·lref         [m]
y_bot = -(H_flu / 2 + n_solid * dx)
y_top =  (H_flu / 2 + n_solid * dx)

# ─── Particle lattice ────────────────────────────────────────────────────────
x_arr = np.linspace(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx, endpoint=True)
y_arr = np.linspace(y_bot + dx / 2,   y_top - dx / 2,   ny, endpoint=True)
z_arr = np.linspace(-lz / 2 + dx / 2, lz / 2 - dx / 2, nz, endpoint=True)

xg, yg, zg = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
positions = np.stack((xg.ravel(), yg.ravel(), zg.ravel()), axis=1).astype(np.float32)

n_particles = len(positions)

# Default: all 'N' (brine, index 1)
typeids    = np.ones(n_particles, dtype=np.int32)
velocities = np.zeros((n_particles, 3), dtype=np.float32)
masses     = np.full(n_particles, mass2, dtype=np.float32)
slengths   = np.full(n_particles, slength, dtype=np.float32)
densities  = np.full(n_particles, rho02, dtype=np.float32)

xp = positions[:, 0]
yp = positions[:, 1]
zp = positions[:, 2]

# ─── H₂ bubble cube ('W', index 0) ───────────────────────────────────────────
# Centred at x = 0, z = 0; TOP face at y = +H_flu/2 (contact with caprock).
# H₂ is buoyant and presses upward against the top solid wall.
half_a      = a_cube / 2
inside_cube = (
    (np.abs(xp) <= half_a) &
    (yp >= H_flu / 2 - a_cube) &
    (yp <  H_flu / 2) &
    (np.abs(zp) <= half_a)
)
typeids[inside_cube]   = 0    # 'W' — H₂
masses[inside_cube]    = mass1
densities[inside_cube] = rho01

# ─── Solid walls ('S', index 2) ──────────────────────────────────────────────
# Both walls initialised with zero velocity.
# The bottom wall velocity is set at runtime before the shear phase.
solid_mask = (yp < -H_flu / 2) | (yp >= H_flu / 2)
typeids[solid_mask]   = 2
masses[solid_mask]    = mass1
densities[solid_mask] = rho01
# velocities already zero

# ─── GSD snapshot ────────────────────────────────────────────────────────────
snapshot = gsd.hoomd.Frame()
snapshot.configuration.box  = [lx, ly, lz, 0, 0, 0]
snapshot.particles.N         = n_particles
snapshot.particles.types     = ['W', 'N', 'S']
snapshot.particles.typeid    = typeids
snapshot.particles.position  = positions
snapshot.particles.velocity  = velocities
snapshot.particles.mass      = masses
snapshot.particles.slength   = slengths
snapshot.particles.density   = densities

init_filename = f'h2brine_{nx}_{ny}_{nz}_vs_{dx:.6f}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

# ─── Summary ─────────────────────────────────────────────────────────────────
n_bub    = int(np.sum(typeids == 0))
n_bri    = int(np.sum(typeids == 1))
n_sol    = int(np.sum(typeids == 2))
V_cube   = n_bub * dx**3
R_hemi   = (3 * V_cube / (2 * np.pi)) ** (1 / 3)
# Reference dimensionless numbers
sigma_r  = 0.01    # surface tension used in run script  [N/m]
gy_r     = 9.81    # gravity magnitude                   [m/s²]
U_wall_r = 0.01    # shear velocity used in run script   [m/s]
mu_brine = 0.001   # brine viscosity used in run script  [Pa·s]
drho = rho02 - rho01   # density difference (buoyancy)
Bo = drho * gy_r * R_hemi**2 / sigma_r
Ca = mu_brine * U_wall_r / sigma_r
Re = rho02 * U_wall_r * lref / mu_brine

print(f'Created {init_filename}')
print(f'  Total particles   : {n_particles}')
print(f'  H₂ bubble W (cube): {n_bub}  (side a = {a_cube*1e3:.3f} mm, '
      f'≈{int(round(a_cube/dx))} layers per edge)')
print(f'  Brine N           : {n_bri}')
print(f'  Solid S           : {n_sol}  ({n_solid} layers each wall)')
print(f'  Domain (fluid)    : {lx*1e3:.2f} × {H_flu*1e3:.2f} × {lz*1e3:.2f} mm  '
      f'(4·lref × 2·lref × 2·lref)')
print(f'  dx                : {dx*1e6:.1f} µm   (lref/dx = {num_length})')
print(f'  R_target          : {R_drop_target*1e3:.3f} mm  (= H_flu/2 = lref)')
print(f'  R_hemi ≈ {R_hemi*1e3:.3f} mm  (spherical cap, volume-conserving)')
print(f'  Density ratio     : ρ_H2/ρ_brine = {rho01/rho02:.2f}  ({rho01}/{rho02} kg/m³)')
print(f'  Bond number  Bo   = {Bo:.4f}  (σ={sigma_r} N/m, g={gy_r} m/s²)')
print(f'  Capillary    Ca   = {Ca:.4f}  (U_wall={U_wall_r} m/s)')
print(f'  Reynolds     Re   = {Re:.1f}')
