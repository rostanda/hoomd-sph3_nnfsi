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

BCC sphere packing — $100 \times 100 \times 100$ voxel cube, geometry generator.

GEOMETRY DESCRIPTION
--------------------
Body-centred cubic (BCC) lattice:
  $\cdot$ $4 \times 4 \times 4$ = 64 unit cells, lattice constant  a = 25 voxels
  $\cdot$ 2 atoms per unit cell (corner + body centre) → 128 spheres total
  $\cdot$ Sphere radius  R = 10 voxels  (R/a = 0.40, below touching $\sqrt{3}/4 \approx 0.433$)
  $\cdot$ Packing has a percolating pore network in all three directions
  $\cdot$ All domain boundaries are periodic

Computed porosity (voxel count):
  $\phi = 1 - 128 \cdot (4\pi/3) \cdot R^3 / (NX \cdot NY \cdot NZ) \approx 0.464$

Physical defaults  (edit the constants below if needed):
  vsize = 0.001 m → domain 100 mm$^3$, sphere diameter d = 20 mm
  rho0  = 1000 kg/m$^3$,  viscosity = 0.001 Pa$\cdot$s  (water at 20 °C)

Kozeny–Carman permeability estimate:
  $k_{KC} = d^2 \phi^3 / [180 (1-\phi)^2] \approx 7.7 \times 10^{-7}\ \mathrm{m}^2$

Usage:
    python3 create_bcc_geometry.py [vsize_m]
      vsize_m : physical voxel size [m]  (default: 0.001)

Output:
    bcc100_init.gsd   — HOOMD initial-state file (all boundaries periodic)
"""

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, os.path.join(_ROOT, 'helper_modules'))

import numpy as np
import hoomd
from hoomd import sph

# ─── geometry parameters (fixed for this benchmark) ──────────────────────────
NX      = NY = NZ = 100         # voxels per direction
N_CELLS = 4                     # BCC unit cells per direction
A_VOX   = NX // N_CELLS         # = 25 voxels  – unit cell lattice constant
R_VOX   = 10                    # sphere radius in voxels  (R/a = 0.40)
assert NX % N_CELLS == 0, 'NX must be divisible by N_CELLS'

# ─── physical parameters ─────────────────────────────────────────────────────
vsize     = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001  # m per voxel
rho0      = 1000.0    # kg/m³
viscosity = 0.001     # Pa·s
kernel    = 'WendlandC4'

LX, LY, LZ = NX * vsize, NY * vsize, NZ * vsize
d_grain     = 2 * R_VOX * vsize     # sphere diameter [m]

# ─── BCC sphere centres in floating-point voxel coordinates ──────────────────
# Voxel ix has its centre at (ix + 0.5) in voxel units.
# Corner atoms sit at voxel corners (integers); body-centre atoms at half-a offsets.
half_a    = A_VOX / 2.0          # = 12.5 voxels

centres_f = []
for icx in range(N_CELLS):
    for icy in range(N_CELLS):
        for icz in range(N_CELLS):
            cx = icx * A_VOX
            cy = icy * A_VOX
            cz = icz * A_VOX
            centres_f.append((float(cx),             float(cy),             float(cz)))
            centres_f.append((float(cx) + half_a,    float(cy) + half_a,    float(cz) + half_a))

# ─── voxelise: mark voxels inside any sphere as solid ────────────────────────
# Using 'ij' indexing: IX varies slowest, IZ fastest in C-order → consistent
# with the position array built below.
IX, IY, IZ = np.meshgrid(np.arange(NX, dtype=np.float32),
                          np.arange(NY, dtype=np.float32),
                          np.arange(NZ, dtype=np.float32), indexing='ij')
ix_f = IX.ravel() + 0.5   # voxel-centre x in voxel units
iy_f = IY.ravel() + 0.5   # voxel-centre y in voxel units
iz_f = IZ.ravel() + 0.5   # voxel-centre z in voxel units

typeid = np.zeros(NX * NY * NZ, dtype=np.uint8)   # 0 = fluid 'F', 1 = solid 'S'

for (cx, cy, cz) in centres_f:
    dx = np.abs(ix_f - cx)
    dy = np.abs(iy_f - cy)
    dz = np.abs(iz_f - cz)
    # Periodic wrap-around
    dx = np.where(dx > NX / 2.0, NX - dx, dx)
    dy = np.where(dy > NY / 2.0, NY - dy, dy)
    dz = np.where(dz > NZ / 2.0, NZ - dz, dz)
    typeid[dx**2 + dy**2 + dz**2 <= R_VOX**2] = 1

n_fluid = int(np.sum(typeid == 0))
n_solid = int(np.sum(typeid == 1))
phi     = n_fluid / float(NX * NY * NZ)

# Kozeny–Carman permeability estimate
k_KC = d_grain**2 * phi**3 / (180.0 * (1.0 - phi)**2)

if True:   # rank-0 always for singlecore create script
    print('BCC sphere packing  100³ voxel cube')
    print(f'  Unit cell a = {A_VOX} voxels = {A_VOX*vsize*1e3:.1f} mm')
    print(f'  Sphere radius R = {R_VOX} voxels = {R_VOX*vsize*1e3:.1f} mm   (R/a = {R_VOX/A_VOX:.3f})')
    print(f'  Sphere diameter d = {d_grain*1e3:.1f} mm')
    print(f'  Domain: {LX*1e3:.0f} mm × {LY*1e3:.0f} mm × {LZ*1e3:.0f} mm')
    print(f'  128 spheres,  {n_fluid} fluid voxels,  {n_solid} solid voxels')
    print(f'  Porosity φ = {phi:.4f}  '
          f'(analytic ≈ {1 - 128*(4/3)*np.pi*R_VOX**3/(NX*NY*NZ):.4f})')
    print(f'  Kozeny–Carman k_KC = {k_KC:.4e} m²  ({k_KC/9.869e-13:.2f} Darcy)')

# ─── particle positions in physical space ─────────────────────────────────────
# HOOMD box: particle positions in [-L/2, +L/2]
positions = np.column_stack([
    ix_f * vsize - LX / 2.0,
    iy_f * vsize - LY / 2.0,
    iz_f * vsize - LZ / 2.0,
]).astype(np.float32)

N_total  = NX * NY * NZ
slength  = hoomd.sph.kernel.OptimalH[kernel] * vsize
mass     = rho0 * vsize**3

# ─── build HOOMD snapshot & write GSD ────────────────────────────────────────
device   = hoomd.device.CPU(notice_level=1)
sim      = hoomd.Simulation(device=device)
snapshot = hoomd.Snapshot(device.communicator)

snapshot.configuration.box      = [LX, LY, LZ, 0.0, 0.0, 0.0]
snapshot.particles.N             = N_total
snapshot.particles.position[:]  = positions
snapshot.particles.typeid[:]     = typeid
snapshot.particles.types         = ['F', 'S']
snapshot.particles.velocity[:]   = np.zeros((N_total, 3), dtype=np.float32)
snapshot.particles.mass[:]       = np.full(N_total, mass,    dtype=np.float32)
snapshot.particles.slength[:]    = np.full(N_total, slength, dtype=np.float32)
snapshot.particles.density[:]    = np.full(N_total, rho0,    dtype=np.float32)

sim.create_state_from_snapshot(snapshot)

outname = 'bcc100_init.gsd'
hoomd.write.GSD.write(state=sim.state, mode='wb', filename=outname)
print(f'  Written: {outname}  ({N_total} particles)')
