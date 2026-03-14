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

Create initial GSD file for the lid-driven cavity benchmark.

Geometry: square cavity of side lref in the x–y plane.
  - Fluid (type 'F', typeid 0): |x| < lref/2 AND |y| < lref/2
  - Solid (type 'S', typeid 1): |x| >= lref/2 OR |y| >= lref/2
    - Top wall (y > lref/2): vx = lidvel  (lid moving in x)
    - All other walls: stationary

Usage (OptionParser):
    python3 create_input_geometries_ldc.py -n <resolution> -R <reynolds>
"""

import sys, os, math
import numpy as np
import gsd.hoomd
from optparse import OptionParser

# ─── kernel constants via hoomd.sph (no device needed) ───────────────────────
import hoomd
from hoomd import sph

# ─── Options ─────────────────────────────────────────────────────────────────
parser = OptionParser()
parser.add_option("-n", "--resolution", type=int,   dest="resolution", default=100)
parser.add_option("-R", "--reynolds",   type=float, dest="reynolds",   default=100.0)
(options, _) = parser.parse_args()

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = options.resolution
lref       = 1.0            # cavity side length                    [m]
rho0       = 1.0            # rest density                          [kg/m³]
lidvel     = 1.0            # lid velocity                          [m/s]
dx         = lref / num_length
mass       = rho0 * dx**3

kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength

part_rcut  = math.ceil(rcut / dx)
part_depth = math.ceil(1.5 * hoomd.sph.kernel.Kappa[kernel] * rcut / dx)

# Domain: solid layers added in both x and y; thin in z
nx = int(num_length + 2 * part_rcut)
ny = int(num_length + 2 * part_rcut)
nz = int(part_depth)

lx = nx * dx
ly = ny * dx
lz = nz * dx

# ─── Particle positions (half-cell offset ensures dx spacing) ────────────────
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

# ─── Particle type + velocity assignment (vectorised) ────────────────────────
x_pos    = positions[:, 0]
y_pos    = positions[:, 1]
top_wall = y_pos >  0.5 * lref
is_solid = (np.abs(x_pos) >= 0.5 * lref) | (np.abs(y_pos) >= 0.5 * lref)

typeid = np.where(is_solid, 1, 0).astype(np.int32)
velocities[top_wall, 0] = lidvel   # lid moves at lidvel in x

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

init_filename = f'liddrivencavity_{nx}_{ny}_{nz}_vs_{dx}_re_{options.reynolds:.0f}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

n_fluid = int(np.sum(typeid == 0))
n_solid = int(np.sum(typeid == 1))
print(f'Written {init_filename}: {n_fluid} fluid, {n_solid} solid ({n_particles} total)')
print(f'  Re={options.reynolds:.0f},  lid velocity vx = {lidvel} m/s')
