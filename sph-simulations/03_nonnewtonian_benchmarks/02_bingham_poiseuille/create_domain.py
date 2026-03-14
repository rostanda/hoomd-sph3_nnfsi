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

Creates the initial GSD file for the Bingham plane Poiseuille benchmark (02).

Written file:
  bingham_{num_length}_init.gsd — parallel plates, body-force driven

Usage:
    python3 create_domain.py [num_length]
    Default: num_length=20
"""

import sys, os, math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, os.path.join(_ROOT, 'helper_modules'))

import hoomd
from hoomd import sph
import numpy as np
import gsd.hoomd

num_length = int(sys.argv[1]) if len(sys.argv) > 1 else 20

lref      = 0.001
rho0      = 1000.0
dx        = lref / num_length
mass      = rho0 * dx**3

kernel     = 'WendlandC4'
slength_dx = hoomd.sph.kernel.OptimalH[kernel]
rcut_sl    = hoomd.sph.kernel.Kappa[kernel]
slength    = slength_dx * dx
rcut       = rcut_sl * slength


def make_parallel_plates_gsd(filename):
    """Parallel plates, no-slip walls, body-force driven."""
    part_rcut = math.ceil(rcut / dx)
    ny = num_length + 3 * part_rcut
    nz = math.ceil(2.5 * rcut_sl * rcut / dx)
    lx, ly, lz = num_length * dx, ny * dx, nz * dx
    xs = np.linspace(-lx/2 + dx/2, lx/2 - dx/2, num_length)
    ys = np.linspace(-ly/2 + dx/2, ly/2 - dx/2, ny)
    zs = np.linspace(-lz/2 + dx/2, lz/2 - dx/2, nz)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
    pos = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
    N   = pos.shape[0]
    tid = np.where(np.abs(pos[:, 1]) >= 0.5 * lref, 1, 0).astype(np.int32)
    snap = gsd.hoomd.Frame()
    snap.configuration.box  = [lx, ly, lz, 0, 0, 0]
    snap.particles.N        = N
    snap.particles.types    = ['F', 'S']
    snap.particles.typeid   = tid
    snap.particles.position = pos.astype(np.float32)
    snap.particles.velocity = np.zeros((N, 3), dtype=np.float32)
    snap.particles.mass     = np.full(N, mass,    dtype=np.float32)
    snap.particles.slength  = np.full(N, slength, dtype=np.float32)
    snap.particles.density  = np.full(N, rho0,    dtype=np.float32)
    with gsd.hoomd.open(filename, 'w') as f:
        f.append(snap)
    return int(np.sum(tid == 0)), int(np.sum(tid == 1))


gsd_file = f'bingham_{num_length}_init.gsd'
nf, ns = make_parallel_plates_gsd(gsd_file)
print(f'Parallel-plates geometry (num_length={num_length}): {nf} fluid + {ns} solid → {gsd_file}')
