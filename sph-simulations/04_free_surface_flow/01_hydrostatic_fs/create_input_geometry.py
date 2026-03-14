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

Create initial GSD for the hydrostatic free-surface benchmark.

Geometry: fluid column of height lref above a solid bottom plate; the top
is a free surface (no top solid wall).

  - Solid (type 'S', typeid 1): y < y_floor  (bottom plate, part_rcut layers)
  - Fluid (type 'F', typeid 0): y_floor <= y < y_floor + lref
  - Vacuum buffer above the fluid:  y >= y_floor + lref  (no particles placed)

The HOOMD box is enlarged by 2*part_rcut rows above the fluid so that
free-surface particles do not see periodic images across the top boundary.

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
lref       = 0.001          # column height                         [m]
rho0       = 1000.0         # rest density                          [kg/m³]
dx         = lref / num_length
mass       = rho0 * dx**3

kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength

part_rcut  = math.ceil(rcut / dx)
part_depth = math.ceil(2.5 * hoomd.sph.kernel.Kappa[kernel] * rcut / dx)

nx = int(part_depth)
nz = int(part_depth)

# ny breakdown:
#   ny_solid  — bottom solid plate
#   ny_fluid  — actual fluid particles
#   ny_buf    — vacuum gap above free surface (box space, no particles placed)
ny_solid = part_rcut
ny_fluid = num_length
ny_buf   = 2 * part_rcut          # wide enough to suppress periodic images
ny_total = ny_solid + ny_fluid + ny_buf

lx = nx * dx
ly = ny_total * dx
lz = nz * dx

xs = np.linspace(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
ys = np.linspace(-ly / 2 + dx / 2, ly / 2 - dx / 2, ny_total)
zs = np.linspace(-lz / 2 + dx / 2, lz / 2 - dx / 2, nz)

xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
pos_all    = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
y_all      = pos_all[:, 1]

# Boundary y-values in box coordinates
y_floor     = -ly / 2 + ny_solid * dx   # solid/fluid interface
y_fluid_top =  y_floor + ny_fluid * dx  # fluid/vacuum interface

solid_mask = y_all <  (y_floor     + 0.1 * dx)
fluid_mask = (y_all >= (y_floor     - 0.1 * dx)) & (y_all < (y_fluid_top + 0.1 * dx)) & ~solid_mask

keep_mask  = solid_mask | fluid_mask
positions  = pos_all[keep_mask]
n_particles = positions.shape[0]

typeid     = np.where(solid_mask[keep_mask], 1, 0).astype(np.int32)
velocities = np.zeros((n_particles, 3), dtype=np.float32)
masses     = np.full(n_particles, mass,    dtype=np.float32)
slengths   = np.full(n_particles, slength, dtype=np.float32)
densities  = np.full(n_particles, rho0,    dtype=np.float32)

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

init_filename = f'hydrostatic_fs_{nx}_{ny_total}_{nz}_vs_{dx:.6f}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

n_fluid = int(np.sum(typeid == 0))
n_solid = int(np.sum(typeid == 1))
print(f'Written {init_filename}: {n_fluid} fluid, {n_solid} solid ({n_particles} total)')
print(f'  y_floor = {y_floor:.6f} m,  y_fluid_top = {y_fluid_top:.6f} m')
print(f'  vacuum buffer: {ny_buf} rows = {ny_buf * dx * 1e3:.3f} mm above free surface')
