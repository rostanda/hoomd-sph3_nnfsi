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

Two-layer Couette flow — initial geometry creation.

BENCHMARK DESCRIPTION
---------------------
Two immiscible fluid layers of equal height (H/2 each) are sheared between
a stationary bottom wall and a top wall moving at U_wall in the x direction.
The channel height (fluid region) is H = lref.  Three solid particle layers
are added above and below the fluid region for the Adami 2012 no-slip BC.

Fluid layer assignment (y coordinate, measured from box centre):
  'W'  lower half  $y \in (-H/2,\ 0)$     $\mu_1$ = 0.004 Pa$\cdot$s   (more viscous)
  'N'  upper half  $y \in (0,\ H/2)$     $\mu_2$ = 0.001 Pa$\cdot$s   (less viscous)

Solid walls:
  'S'  $y < -H/2$    velocity = (0, 0, 0)       bottom wall, stationary
  'S'  $y >  H/2$    velocity = (U_wall, 0, 0)  top wall, moving

ANALYTICAL SOLUTION
-------------------
With layer interface at y = 0, bottom wall at $y = -H/2$, top wall at $y = H/2$:

  Interface velocity   $v_i = U\_wall \times \mu_2 / (\mu_1 + \mu_2)$

  Lower layer  $v_1(y) = v_i \times (y + H/2) / (H/2)$   for $y \in [-H/2,\ 0]$
  Upper layer  $v_2(y) = v_i + (U\_wall - v_i) \times y / (H/2)$  for $y \in [0,\ H/2]$

Default parameters ($\mu_1$=0.004, $\mu_2$=0.001):
  $v_i = U\_wall \times 0.001 / 0.005 = 0.2 \times U\_wall$
  Lower shear rate $= 0.4 \times U\_wall / H$
  Upper shear rate $= 1.6 \times U\_wall / H$

Usage:
    python3 create_input_geometry.py <num_length>

    num_length : integer resolution — particles across H = lref (e.g. 20, 40)

Output:
    couette_<nx>_<ny>_<nz>_vs_<dx>_init.gsd
"""

import sys
import numpy as np
import gsd.hoomd
import hoomd
from hoomd import sph

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = int(sys.argv[1]) if len(sys.argv) > 1 else 20

lref    = 0.001                    # channel height = reference length  [m]
U_wall  = 0.01                     # top wall velocity                  [m/s]
rho0    = 1000.0                   # rest density (both phases)         [kg/m³]
dx      = lref / num_length        # particle spacing                   [m]
mass    = rho0 * dx**3             # particle mass                      [kg]
n_solid = 3                        # solid particle layers (top+bottom)

kernel  = 'WendlandC4'
slength = hoomd.sph.kernel.OptimalH[kernel] * dx   # smoothing length  [m]

# ─── Box geometry ────────────────────────────────────────────────────────────
# Fluid region: $l_\mathrm{ref} \times l_\mathrm{ref} \times l_\mathrm{ref}$.
# Wall region:  n_solid layers on each side in y → total ny adds 2*n_solid.
# Periodic in x and z; solid walls provide the y boundaries.
nx = num_length
ny = num_length + 2 * n_solid
nz = num_length
lx = nx * dx
ly = ny * dx
lz = nz * dx

H     = num_length * dx            # fluid channel height = lref        [m]
y_bot = -(H / 2 + n_solid * dx)   # bottom of the box (HOOMD centre)   [m]
y_top =  (H / 2 + n_solid * dx)   # top of the box                     [m]

# ─── Particle lattice ────────────────────────────────────────────────────────
x_arr = np.linspace(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx, endpoint=True)
y_arr = np.linspace(y_bot + dx / 2,   y_top - dx / 2,   ny, endpoint=True)
z_arr = np.linspace(-lz / 2 + dx / 2, lz / 2 - dx / 2, nz, endpoint=True)

xg, yg, zg = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
positions = np.stack((xg.ravel(), yg.ravel(), zg.ravel()), axis=1).astype(np.float32)

n_particles = len(positions)
typeids    = np.zeros(n_particles, dtype=np.int32)   # default: 'W' (index 0)
velocities = np.zeros((n_particles, 3), dtype=np.float32)
masses     = np.full(n_particles, mass, dtype=np.float32)
slengths   = np.full(n_particles, slength, dtype=np.float32)
densities  = np.full(n_particles, rho0, dtype=np.float32)

y = positions[:, 1]

# Upper fluid layer ('N', index 1): y ∈ (0, H/2)
upper_fluid = (y >= 0.0) & (y < H / 2)
typeids[upper_fluid] = 1

# Solid bottom wall ('S', index 2): y < −H/2 — stationary
solid_bot = y < -H / 2
typeids[solid_bot]   = 2
velocities[solid_bot] = [0.0, 0.0, 0.0]

# Solid top wall ('S', index 2): y >= H/2 — moving at U_wall in x
solid_top = y >= H / 2
typeids[solid_top]        = 2
velocities[solid_top, 0]  = U_wall   # vx = U_wall

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

init_filename = f'couette_{nx}_{ny}_{nz}_vs_{dx:.6f}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

# Analytical interface velocity
mu1   = 0.004
mu2   = 0.001
v_i   = U_wall * mu2 / (mu1 + mu2)

print(f'Created {init_filename}')
print(f'  Total particles  : {n_particles}')
print(f'  Fluid W (lower)  : {int(np.sum((typeids==0)))} particles, μ₁={mu1} Pa·s')
print(f'  Fluid N (upper)  : {int(np.sum((typeids==1)))} particles, μ₂={mu2} Pa·s')
print(f'  Solid S          : {int(np.sum((typeids==2)))} particles')
print(f'  Channel height H : {H*1e3:.3f} mm')
print(f'  U_wall           : {U_wall:.4f} m/s')
print(f'  Analytical v_i   : {v_i:.4f} m/s  ({v_i/U_wall*100:.0f}% of U_wall)')
