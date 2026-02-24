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

Create initial GSD file for the 2-D Taylor–Green vortex benchmark.

Geometry: fully periodic square domain [0, L]^2 (thin in z).
All particles are fluid.  Initial velocities follow the Taylor–Green pattern:

    v_x(x, y) = -U0 * cos(2π x/L) * sin(2π y/L)
    v_y(x, y) =  U0 * sin(2π x/L) * cos(2π y/L)

The analytical solution decays as exp(-2 ν k^2 t)  with k = 2π/L.

Usage:
    python3 create_input_geometry.py <num_length> [U0]
"""

import sys, math
import numpy as np
import gsd.hoomd

import hoomd
from hoomd import sph

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = int(sys.argv[1])
U0         = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
lref       = 1.0            # domain side length                    [m]
rho0       = 1.0            # rest density                          [kg/m³]
dx         = lref / num_length
mass       = rho0 * dx**3

kernel     = 'WendlandC4'
slength    = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut       = hoomd.sph.kernel.Kappa[kernel] * slength

part_depth = max(3, math.ceil(2.0 * rcut / dx))

# Square domain, thin in z (quasi-2D)
nx = int(num_length)
ny = int(num_length)
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

# ─── Initial Taylor–Green velocity field ─────────────────────────────────────
k      = 2.0 * np.pi / lx  # wavenumber
x_pos  = positions[:, 0]
y_pos  = positions[:, 1]

velocities = np.zeros((n_particles, 3), dtype=np.float32)
velocities[:, 0] = (-U0 * np.cos(k * x_pos) * np.sin(k * y_pos)).astype(np.float32)
velocities[:, 1] = ( U0 * np.sin(k * x_pos) * np.cos(k * y_pos)).astype(np.float32)

masses   = np.full(n_particles, mass,    dtype=np.float32)
slengths = np.full(n_particles, slength, dtype=np.float32)
densities = np.full(n_particles, rho0,   dtype=np.float32)
typeid   = np.zeros(n_particles, dtype=np.int32)  # all fluid

snapshot = gsd.hoomd.Frame()
snapshot.configuration.box  = [lx, ly, lz, 0, 0, 0]
snapshot.particles.N        = n_particles
snapshot.particles.types    = ['F', 'S']  # 'S' listed but unused (for compatibility)
snapshot.particles.typeid   = typeid
snapshot.particles.position = positions.astype(np.float32)
snapshot.particles.velocity = velocities
snapshot.particles.mass     = masses
snapshot.particles.slength  = slengths
snapshot.particles.density  = densities

init_filename = f'taylor_green_{nx}_{ny}_{nz}_vs_{dx}_U0_{U0}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

print(f'Written {init_filename}: {n_particles} fluid particles')
print(f'  L={lx:.3f} m,  k={k:.4f} rad/m,  U0={U0} m/s')
print(f'  Decay time scale t* = L^2 / (4*pi^2*nu)  (nu-dependent)')
