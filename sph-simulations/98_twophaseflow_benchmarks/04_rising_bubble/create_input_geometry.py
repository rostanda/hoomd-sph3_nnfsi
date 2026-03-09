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

Rising bubble — initial geometry creation.

BENCHMARK DESCRIPTION
---------------------
A spherical bubble of phase 'N' (gas, $\rho_2$ = 100 kg/m$^3$) is placed at rest at
the bottom quarter of a tall rectangular domain filled with liquid 'W'
($\rho_1$ = 1000 kg/m$^3$).  Gravity acts in the $-y$ direction.  Solid walls bound the
domain in y; x and z are periodic.

Domain: $lref \times 2 \cdot lref \times lref$  (width $\times$ height $\times$ depth)
Bubble: radius R = 0.2 $\times$ lref, centre at $y = -lref/2$

Key dimensionless numbers (default parameters):
  Eötvös   $Eo = (\rho_1 - \rho_2) g D^2 / \sigma$
             $= 900 \times 9.81 \times (0.4 \times 10^{-3})^2 / 0.05 \approx 28.2$
  Morton   $Mo = g \mu_1^4 (\rho_1 - \rho_2) / (\rho_1^2 \sigma^3)$
             $\approx 1.27 \times 10^{-8}$
  Reynolds $Re = \rho_1 U_T D / \mu_1$   (U_T estimated from Hadamard–Rybczynski)

Solid walls bound $y = \pm l_\mathrm{ref}$.  Three particle layers are used for each wall.

Usage:
    python3 create_input_geometry.py <num_length>

    num_length : integer resolution — particles across lref (e.g. 20, 40)

Output:
    rising_bubble_<nx>_<ny>_<nz>_vs_<dx>_init.gsd
"""

import sys
import numpy as np
import gsd.hoomd
import hoomd
from hoomd import sph

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = int(sys.argv[1]) if len(sys.argv) > 1 else 20

lref    = 0.001                    # reference length (domain width)    [m]
R_bub   = 0.2 * lref               # bubble radius                      [m]
rho01   = 1000.0                   # rest density liquid (outer, 'W')   [kg/m³]
rho02   = 100.0                    # rest density gas    (bubble, 'N')  [kg/m³]
dx      = lref / num_length        # particle spacing                   [m]
mass1   = rho01 * dx**3            # particle mass phase W              [kg]
mass2   = rho02 * dx**3            # particle mass phase N              [kg]
n_solid = 3                        # solid wall layers on each side

kernel  = 'WendlandC4'
slength = hoomd.sph.kernel.OptimalH[kernel] * dx

# ─── Box geometry ────────────────────────────────────────────────────────────
# Fluid region: $l_\mathrm{ref}$ (x) $\times$ $2 \cdot l_\mathrm{ref}$ (y) $\times$ $l_\mathrm{ref}$ (z)
# Solid layers: n_solid rows added top and bottom in y.
nx     = num_length
ny_flu = 2 * num_length            # fluid rows in y
ny     = ny_flu + 2 * n_solid      # total rows including walls
nz     = num_length
lx = nx * dx
ly = ny * dx
lz = nz * dx

H_flu = ny_flu * dx                # fluid height = 2*lref              [m]
y_bot = -(H_flu / 2 + n_solid * dx)
y_top =  (H_flu / 2 + n_solid * dx)

# ─── Particle lattice ────────────────────────────────────────────────────────
x_arr = np.linspace(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx, endpoint=True)
y_arr = np.linspace(y_bot + dx / 2,   y_top - dx / 2,   ny, endpoint=True)
z_arr = np.linspace(-lz / 2 + dx / 2, lz / 2 - dx / 2, nz, endpoint=True)

xg, yg, zg = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
positions = np.stack((xg.ravel(), yg.ravel(), zg.ravel()), axis=1).astype(np.float32)

n_particles = len(positions)
typeids    = np.zeros(n_particles, dtype=np.int32)    # default: 'W'
velocities = np.zeros((n_particles, 3), dtype=np.float32)
masses     = np.full(n_particles, mass1, dtype=np.float32)
slengths   = np.full(n_particles, slength, dtype=np.float32)
densities  = np.full(n_particles, rho01, dtype=np.float32)

y = positions[:, 1]
x = positions[:, 0]
z = positions[:, 2]

# Bubble centre at y = −lref/2 (bottom quarter of fluid region)
y_bub = -lref / 2
r_sq  = x**2 + (y - y_bub)**2 + z**2
inside_bubble = r_sq < R_bub**2

typeids[inside_bubble]   = 1     # 'N' — gas bubble
masses[inside_bubble]    = mass2
densities[inside_bubble] = rho02

# Solid walls ('S', index 2)
solid_bot = y < -H_flu / 2
solid_top = y >=  H_flu / 2
typeids[solid_bot | solid_top] = 2

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

init_filename = f'rising_bubble_{nx}_{ny}_{nz}_vs_{dx:.6f}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

sigma = 0.05
gy    = 9.81
D     = 2 * R_bub
Eo    = (rho01 - rho02) * gy * D**2 / sigma

print(f'Created {init_filename}')
print(f'  Total particles  : {n_particles}')
print(f'  Liquid W         : {int(np.sum(typeids==0))} particles, ρ={rho01} kg/m³')
print(f'  Bubble N         : {int(np.sum(typeids==1))} particles, ρ={rho02} kg/m³')
print(f'  Solid S          : {int(np.sum(typeids==2))} particles')
print(f'  Bubble radius    : R = {R_bub*1e3:.3f} mm')
print(f'  Density ratio    : {rho01/rho02:.0f}:1')
print(f'  Eötvös number    : Eo = {Eo:.1f}  (for σ=0.05 N/m, g=9.81 m/s²)')
