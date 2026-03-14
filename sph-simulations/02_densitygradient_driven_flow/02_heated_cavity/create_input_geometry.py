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

Create initial GSD file for the differentially heated square cavity benchmark.

GEOMETRY
--------
Square cavity of side $L \times L$ enclosed by four solid walls.
  - Fluid (type 'F', typeid 0): |x| < L/2  AND  |y| < L/2
  - Solid (type 'S', typeid 1): all remaining particles (wall layers)

Wall temperatures are assigned in the run script from particle position:
  - Left wall  (x < −L/2): hot,  T_hot
  - Right wall (x >  L/2): cold, T_cold
  - Top/bottom walls ($|y| > L/2$): $T_\mathrm{avg} \approx$ adiabatic

The domain is periodic in z (thin quasi-2D slice).

Reference: de Vahl Davis (1983), Int. J. Num. Meth. Fluids 3, 249–264.

Usage:
    python3 create_input_geometry.py <num_length>

    num_length : number of fluid particles across the cavity length L
"""

import sys, os, math
import numpy as np
import gsd.hoomd

import hoomd
from hoomd import sph

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = int(sys.argv[1])

L       = 1.0          # cavity side length                  [m]
rho0    = 1.0          # rest density                        [kg/m³]
dx      = L / num_length
mass    = rho0 * dx**3

kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength

part_rcut  = math.ceil(rcut / dx)
part_depth = math.ceil(2.5 * hoomd.sph.kernel.Kappa[kernel] * rcut / dx)  # $\ell_z > 2\kappa h$ (pseudo-3D: more than one kernel diameter in z)

# Domain: periodic in z (thin slice), solid wall layers in x and y
nx = int(num_length + 2 * part_rcut)   # fluid + left/right wall layers
ny = int(num_length + 2 * part_rcut)   # fluid + top/bottom wall layers
nz = int(part_depth)

lx = nx * dx
ly = ny * dx
lz = nz * dx

# ─── Particle positions ──────────────────────────────────────────────────────
xs = np.linspace(-lx / 2, lx / 2, nx, endpoint=True)
ys = np.linspace(-ly / 2, ly / 2, ny, endpoint=True)
zs = np.linspace(-lz / 2, lz / 2, nz, endpoint=True)

xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
positions   = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
n_particles = positions.shape[0]

velocities = np.zeros((n_particles, 3), dtype=np.float32)
masses     = np.full(n_particles, mass,    dtype=np.float32)
slengths   = np.full(n_particles, slength, dtype=np.float32)
densities  = np.full(n_particles, rho0,    dtype=np.float32)

# ─── Type assignment (vectorised): fluid inside cavity, solid outside ─────────
x_pos    = positions[:, 0]
y_pos    = positions[:, 1]
is_solid = (x_pos < -0.5 * L) | (x_pos > 0.5 * L) | \
           (y_pos < -0.5 * L) | (y_pos > 0.5 * L)

typeid = np.where(is_solid, 1, 0).astype(np.int32)

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

init_filename = f'heated_cavity_{num_length}x{num_length}_dx_{dx:.4f}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

n_fluid = int(np.sum(typeid == 0))
n_solid = int(np.sum(typeid == 1))
print(f'Written {init_filename}: {n_fluid} fluid, {n_solid} solid ({n_particles} total)')
print(f'  Cavity L = {L:.2f} m,  dx = {dx:.4f} m,  num_length = {num_length}')
print(f'  Box: {lx:.3f} × {ly:.3f} × {lz:.4f} m  (nx={nx}, ny={ny}, nz={nz})')
print(f'  Wall temperatures assigned in run script from particle position.')
