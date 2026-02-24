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

Create initial GSD for the 2-D dam-break benchmark.

Geometry (x = horizontal, y = vertical, z = thin periodic slab):
  - Column width:    a   = lref
  - Column height:   H₀  = 2 * a
  - Channel length:  L   = 4 * a
  - Solid floor:  y < 0      (part_rcut layers, full channel width)
  - Solid left:   x < 0      (part_rcut layers, full height)
  - Solid right:  x > L      (part_rcut layers, full height)
  - Fluid:        0 ≤ x ≤ a,  0 ≤ y ≤ H₀   (initial water column)
  - Vacuum:       x > a  or  y > H₀          (no particles — free to flood)

Box sized to include solid wall layers and a vacuum buffer above H₀.

Usage:
    python3 create_input_geometry.py <num_length>
    e.g.  python3 create_input_geometry.py 20
"""

import sys, math
import numpy as np
import gsd.hoomd

import hoomd
from hoomd import sph

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = int(sys.argv[1])        # particles per reference length a
lref       = 0.01                    # dam-column width a             [m]
H0         = 2.0 * lref             # initial column height          [m]
L_channel  = 4.0 * lref             # total channel length           [m]
rho0       = 1000.0                  # rest density                   [kg/m³]
dx         = lref / num_length
mass       = rho0 * dx**3

kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength

n_s   = math.ceil(rcut / dx)        # solid wall layers
n_buf = 2 * n_s                     # vacuum buffer above H₀
nz    = max(3, math.ceil(2.5 * hoomd.sph.kernel.Kappa[kernel] * rcut / dx))

# ─── Grid in physical coordinates ────────────────────────────────────────────
# x: from -n_s*dx to L_channel + n_s*dx
# y: from -n_s*dx to H0 + n_buf*dx
# z: thin slab centred at 0
x_vals = np.arange(-(n_s - 0.5) * dx, L_channel + n_s * dx, dx)
y_vals = np.arange(-(n_s - 0.5) * dx, H0 + n_buf * dx,      dx)
z_vals = np.arange(-(nz / 2) * dx,    (nz / 2) * dx,         dx)

# HOOMD box dimensions (centred at origin)
lx = x_vals[-1] - x_vals[0] + dx
ly = y_vals[-1] - y_vals[0] + dx
lz = z_vals[-1] - z_vals[0] + dx

# Shift to box-centred coordinates
x_vals -= (x_vals[0] + x_vals[-1]) / 2
y_vals -= (y_vals[0] + y_vals[-1]) / 2
z_vals -= (z_vals[0] + z_vals[-1]) / 2

xg, yg, zg = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
pos_all = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
x_all   = pos_all[:, 0]
y_all   = pos_all[:, 1]

# Boundary values in box coordinates
#   physical x=0 corresponds to the left solid/fluid boundary
#   physical y=0 corresponds to the floor solid/fluid boundary
x_left   = x_vals[n_s - 1] + dx / 2    # left  solid | fluid boundary
x_right  = x_left + L_channel           # right fluid | solid boundary
x_col    = x_left + lref                # right edge of initial column
y_floor  = y_vals[n_s - 1] + dx / 2    # bottom solid | fluid boundary
y_col    = y_floor + H0                 # top of initial fluid column

solid_mask = (
    (y_all < y_floor)                                                  # floor
    | ((x_all < x_left)  & (y_all >= y_floor - 0.1 * dx))            # left wall
    | ((x_all >= x_right) & (y_all >= y_floor - 0.1 * dx))           # right wall
)
fluid_mask = (
    (~solid_mask)
    & (x_all >= x_left  - 0.1 * dx) & (x_all < x_col   + 0.1 * dx)
    & (y_all >= y_floor - 0.1 * dx) & (y_all < y_col   + 0.1 * dx)
)

keep_mask   = solid_mask | fluid_mask
positions   = pos_all[keep_mask]
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

nx_grid = len(x_vals)
ny_grid = len(y_vals)
init_filename = f'dam_break_{nx_grid}_{ny_grid}_{nz}_vs_{dx:.6f}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

n_fluid = int(np.sum(typeid == 0))
n_solid = int(np.sum(typeid == 1))
print(f'Written {init_filename}: {n_fluid} fluid, {n_solid} solid ({n_particles} total)')
print(f'  a = {lref*1e3:.1f} mm,  H₀ = {H0*1e3:.1f} mm,  L = {L_channel*1e3:.1f} mm')
print(f'  dx = {dx*1e3:.3f} mm,  slength = {slength*1e3:.3f} mm,  n_s = {n_s}')
