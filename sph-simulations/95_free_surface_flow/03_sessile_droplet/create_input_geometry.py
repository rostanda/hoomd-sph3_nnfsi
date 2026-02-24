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

Create initial GSD for the sessile-droplet contact-angle benchmark.

A 2-D liquid droplet (periodic in x, thin slab in z) is initialised as a
semicircle (θ_init = 90°) resting on a flat solid wall.  Surface tension
and contact-angle enforcement (SinglePhaseFlowFS) drive it toward the
prescribed equilibrium contact angle θ_eq.

Geometry:
  - Flat solid wall: y < y_floor  (n_s layers of solid particles)
  - Fluid droplet:   x² + (y - y_floor)² < R_drop²,  y > y_floor
    (semicircular initial shape; θ_init = 90°)
  - Vacuum above and around the droplet (no particles placed)
  - Domain in x is periodic; width = 4·R_drop so the droplet is isolated.

Volume conservation: the semicircle area π·R²/2 is conserved as the
droplet relaxes.  The equilibrium cap radius and height for angle θ_eq:
    A_cap = R_cap² (θ_eq − sin θ_eq cos θ_eq)  =  π R_drop² / 2
    r_base = R_cap sin θ_eq,   h_cap = R_cap (1 − cos θ_eq)
    contact angle from final shape:  θ = 2 atan(h / r_base)

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
num_length = int(sys.argv[1])        # particles across R_drop
lref       = 0.001                   # reference length               [m]
R_drop     = 0.4 * lref             # initial semicircle radius      [m]
rho0       = 1000.0                  # rest density                   [kg/m³]
dx         = R_drop / num_length
mass       = rho0 * dx**3

kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength

n_s    = math.ceil(rcut / dx)       # solid wall layers
n_buf  = 3 * n_s                    # vacuum buffer above droplet
nz     = max(3, math.ceil(2.5 * hoomd.sph.kernel.Kappa[kernel] * rcut / dx))

# ─── Domain ──────────────────────────────────────────────────────────────────
# x: periodic, width = 4*R_drop (droplet centred, enough space around it)
# y: solid floor (n_s layers) + droplet height R_drop + vacuum (n_buf layers)
Lx = 4.0 * R_drop
Ly_solid = n_s * dx
Ly_fluid = R_drop            # maximum droplet height (semicircle top)
Ly_buf   = n_buf * dx
Ly       = Ly_solid + Ly_fluid + Ly_buf
Lz       = nz * dx

# Grid in box coordinates (origin at box centre)
xs = np.arange(-Lx / 2 + dx / 2, Lx / 2, dx)
ys = np.arange(-Ly / 2 + dx / 2, Ly / 2, dx)
zs = np.arange(-Lz / 2 + dx / 2, Lz / 2, dx)

xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
pos_all = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
x_all   = pos_all[:, 0]
y_all   = pos_all[:, 1]

# Floor boundary in box coordinates
y_floor = -Ly / 2 + Ly_solid   # solid/fluid interface

solid_mask = y_all < (y_floor + 0.1 * dx)

# Fluid: semicircle x² + (y - y_floor)² < R_drop², y > y_floor
r_sq       = x_all**2 + (y_all - y_floor)**2
fluid_mask = (~solid_mask) & (r_sq < (R_drop - 0.1 * dx)**2)

keep_mask   = solid_mask | fluid_mask
positions   = pos_all[keep_mask]
n_particles = positions.shape[0]

typeid     = np.where(solid_mask[keep_mask], 1, 0).astype(np.int32)
velocities = np.zeros((n_particles, 3), dtype=np.float32)
masses     = np.full(n_particles, mass,    dtype=np.float32)
slengths   = np.full(n_particles, slength, dtype=np.float32)
densities  = np.full(n_particles, rho0,    dtype=np.float32)

snapshot = gsd.hoomd.Frame()
snapshot.configuration.box  = [Lx, Ly, Lz, 0, 0, 0]
snapshot.particles.N        = n_particles
snapshot.particles.types    = ['F', 'S']
snapshot.particles.typeid   = typeid
snapshot.particles.position = positions.astype(np.float32)
snapshot.particles.velocity = velocities
snapshot.particles.mass     = masses
snapshot.particles.slength  = slengths
snapshot.particles.density  = densities

nx_g = len(xs)
ny_g = len(ys)
init_filename = f'sessile_droplet_{nx_g}_{ny_g}_{nz}_vs_{dx:.6f}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

n_fluid = int(np.sum(typeid == 0))
n_solid = int(np.sum(typeid == 1))
print(f'Written {init_filename}: {n_fluid} fluid, {n_solid} solid ({n_particles} total)')
print(f'  R_drop = {R_drop*1e3:.2f} mm,  dx = {dx*1e6:.1f} µm,  n_s = {n_s}')
print(f'  Initial shape: semicircle, θ_init = 90°')
