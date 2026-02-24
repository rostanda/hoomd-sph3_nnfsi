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

Create initial GSD file for Stokes' second problem (oscillating plate).

Geometry: fluid layer of height H = lref above an oscillating solid plate at
y = -H/2.  The top boundary is a stationary solid plate at y = +H/2
(chosen so that H >> delta, where delta = sqrt(2*nu/omega) is the Stokes
layer thickness — the upper plate should have negligible influence).

  - Fluid (type 'F', typeid 0): |y| < H/2
  - Solid (type 'S', typeid 1): |y| >= H/2
    - Bottom plate (y < -H/2): will be oscillated by run script
    - Top plate (y > H/2): stationary

Usage:
    python3 create_input_geometry.py <num_length>
"""

import sys, math
import numpy as np
import gsd.hoomd

import hoomd
from hoomd import sph

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = int(sys.argv[1])
lref       = 0.001          # fluid layer height H                  [m]
rho0       = 1000.0         # rest density                          [kg/m³]
dx         = lref / num_length
mass       = rho0 * dx**3

kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength

part_rcut  = math.ceil(rcut / dx)
part_depth = math.ceil(2.5 * hoomd.sph.kernel.Kappa[kernel] * rcut / dx)

# Domain: periodic in x (and z), solid layers in y
nx = int(part_depth)
ny = int(num_length + 3 * part_rcut)
nz = int(part_depth)

lx = nx * dx
ly = ny * dx
lz = nz * dx

xs = np.linspace(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx, endpoint=True)
ys = np.linspace(-ly / 2 + dx / 2, ly / 2 - dx / 2, ny, endpoint=True)
zs = np.linspace(-lz / 2 + dx / 2, lz / 2 - dx / 2, nz, endpoint=True)

xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
positions  = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
n_particles = positions.shape[0]

velocities = np.zeros((n_particles, 3), dtype=np.float32)
masses     = np.full(n_particles, mass,    dtype=np.float32)
slengths   = np.full(n_particles, slength, dtype=np.float32)
densities  = np.full(n_particles, rho0,    dtype=np.float32)

y_pos  = positions[:, 1]
typeid = np.where(np.abs(y_pos) >= 0.5 * lref, 1, 0).astype(np.int32)

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

init_filename = f'stokes_second_{nx}_{ny}_{nz}_vs_{dx}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

n_fluid = int(np.sum(typeid == 0))
n_solid = int(np.sum(typeid == 1))
print(f'Written {init_filename}: {n_fluid} fluid, {n_solid} solid ({n_particles} total)')
