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

Static droplet — initial geometry creation.

BENCHMARK DESCRIPTION
---------------------
A spherical droplet of radius R = 0.25 × lref (phase 'B') is placed at rest in
a periodic box filled with an equal-density outer fluid (phase 'A').  There are
no solid walls and no gravity.

After the simulation equilibrates, the pressure inside the droplet exceeds the
pressure outside by the Young–Laplace pressure:

    ΔP_theory = 2σ / R     [3-D sphere, N/m → Pa]

with default parameters (σ = 0.01 N/m, R = 0.25 × 0.001 m):
    ΔP_theory = 2 × 0.01 / 2.5e-4 = 80 Pa

Spurious velocities (parasitic currents) arise from numerical discretisation of
the surface tension gradient.  Their magnitude should be checked against the
capillary velocity scale  U_cap = σ / μ  (here 10 m/s) and should be well below
1 % of U_cap for a good solver.  The TV variant (run_static_droplet_TV.py)
generally achieves an order-of-magnitude lower spurious currents.

Usage:
    python3 create_input_geometry.py <num_length>

    num_length : integer resolution — particles across lref (e.g. 20, 40)

Output:
    static_droplet_<nx>_<ny>_<nz>_vs_<dx>_init.gsd
"""

import sys, os
import numpy as np
import math
import gsd.hoomd
import hoomd
from hoomd import sph

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = int(sys.argv[1]) if len(sys.argv) > 1 else 20

lref     = 0.001                    # domain / reference length      [m]
R_drop   = 0.25 * lref             # droplet radius                 [m]
rho01    = 1000.0                   # rest density phase A (outer)   [kg/m³]
rho02    = 1000.0                   # rest density phase B (droplet) [kg/m³]
dx       = lref / num_length        # particle spacing               [m]
mass1    = rho01 * dx**3            # particle mass phase A          [kg]
mass2    = rho02 * dx**3            # particle mass phase B          [kg]

kernel   = 'WendlandC4'
slength  = hoomd.sph.kernel.OptimalH[kernel] * dx   # smoothing length [m]
rcut     = hoomd.sph.kernel.Kappa[kernel] * slength  # cutoff radius   [m]

# ─── Box geometry (full periodic, no walls) ──────────────────────────────────
# The box is lref × lref × lref (4R on each side).
# Particles are placed on a regular lattice that spans the full box.
nx, ny, nz = num_length, num_length, num_length
lx = nx * dx
ly = ny * dx
lz = nz * dx

n_particles = nx * ny * nz

x, y, z = np.meshgrid(
    np.linspace(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx, endpoint=True),
    np.linspace(-ly / 2 + dx / 2, ly / 2 - dx / 2, ny, endpoint=True),
    np.linspace(-lz / 2 + dx / 2, lz / 2 - dx / 2, nz, endpoint=True),
    indexing='ij')

positions  = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1).astype(np.float32)
velocities = np.zeros((n_particles, 3), dtype=np.float32)
masses     = np.full(n_particles, mass1, dtype=np.float32)
slengths   = np.full(n_particles, slength, dtype=np.float32)
densities  = np.full(n_particles, rho01, dtype=np.float32)
typeids    = np.zeros(n_particles, dtype=np.int32)   # 0 = 'A' (outer fluid)

# ─── Assign droplet particles (sphere of radius R_drop at origin) ─────────────
r_sq = positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2
inside_droplet = r_sq < R_drop**2
typeids[inside_droplet]    = 1          # 'B' — inner droplet phase
masses[inside_droplet]     = mass2
densities[inside_droplet]  = rho02

# ─── GSD snapshot ────────────────────────────────────────────────────────────
snapshot = gsd.hoomd.Frame()
snapshot.configuration.box = [lx, ly, lz, 0, 0, 0]
snapshot.particles.N        = n_particles
snapshot.particles.types    = ['A', 'B']
snapshot.particles.typeid   = typeids
snapshot.particles.position  = positions
snapshot.particles.velocity  = velocities
snapshot.particles.mass      = masses
snapshot.particles.slength   = slengths
snapshot.particles.density   = densities

init_filename = f'static_droplet_{nx}_{ny}_{nz}_vs_{dx:.6f}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

print(f'Created {init_filename}')
print(f'  Total particles : {n_particles}')
print(f'  Droplet particles: {int(inside_droplet.sum())}')
print(f'  Outer particles  : {int((~inside_droplet).sum())}')
print(f'  Droplet radius   : R = {R_drop*1e3:.3f} mm')
print(f'  dx               : {dx*1e6:.1f} µm   (R/dx = {R_drop/dx:.1f})')
print(f'  Expected ΔP      : 2σ/R = 80 Pa  (for σ=0.01 N/m)')
