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

Creates the initial GSD file for the Bingham slump test benchmark (04).

Written file:
  slump_{num_length}_init.gsd — fluid column (H0 x 2L0) on a solid floor

Layout (y=0 at floor surface):
  Fluid: x ∈ [-L0, L0], y ∈ [0, H0]
  Solid: full box width, n_solid layers below y=0

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

lref      = 0.10
H0        = lref
L0        = 0.05
rho0      = 2200.0
x_box     = 0.70
dx        = lref / num_length
mass      = rho0 * dx**3

kernel     = 'WendlandC4'
slength_dx = hoomd.sph.kernel.OptimalH[kernel]
rcut_sl    = hoomd.sph.kernel.Kappa[kernel]
slength    = slength_dx * dx
rcut       = rcut_sl * slength


def make_slump_gsd(filename):
    """Fluid column resting on a solid floor."""
    n_solid = math.ceil(rcut / dx) + 1
    nz      = math.ceil(2.5 * rcut_sl * rcut / dx)
    nx_box  = int(round(x_box / dx))
    ly      = 0.40
    lx      = nx_box * dx
    lz      = nz * dx

    xs_all   = np.linspace(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx_box)
    nx_fluid = int(round(2.0 * L0 / dx))
    ny_fluid = num_length
    xs_fluid = np.linspace(-L0 + dx / 2, L0 - dx / 2, nx_fluid)
    ys_fluid = np.linspace(dx / 2, H0 - dx / 2, ny_fluid)
    ys_solid = np.array([-(i + 0.5) * dx for i in range(n_solid)])
    zs       = np.linspace(-lz / 2 + dx / 2, lz / 2 - dx / 2, nz)

    xg_f, yg_f, zg_f = np.meshgrid(xs_fluid, ys_fluid, zs, indexing='ij')
    pos_f = np.column_stack([xg_f.ravel(), yg_f.ravel(), zg_f.ravel()])

    xg_s, yg_s, zg_s = np.meshgrid(xs_all, ys_solid, zs, indexing='ij')
    pos_s = np.column_stack([xg_s.ravel(), yg_s.ravel(), zg_s.ravel()])

    pos_all = np.vstack([pos_f, pos_s]).astype(np.float32)
    N       = pos_all.shape[0]
    tid     = np.zeros(N, dtype=np.int32)
    tid[len(pos_f):] = 1   # solid = type 1

    snap = gsd.hoomd.Frame()
    snap.configuration.box  = [lx, ly, lz, 0, 0, 0]
    snap.particles.N        = N
    snap.particles.types    = ['F', 'S']
    snap.particles.typeid   = tid
    snap.particles.position = pos_all
    snap.particles.velocity = np.zeros((N, 3), dtype=np.float32)
    snap.particles.mass     = np.full(N, mass,    dtype=np.float32)
    snap.particles.slength  = np.full(N, slength, dtype=np.float32)
    snap.particles.density  = np.full(N, rho0,    dtype=np.float32)

    with gsd.hoomd.open(filename, 'w') as f:
        f.append(snap)

    return int(np.sum(tid == 0)), int(np.sum(tid == 1))


gsd_file = f'slump_{num_length}_init.gsd'
nf, ns = make_slump_gsd(gsd_file)
print(f'Slump geometry (num_length={num_length}): dx={dx*1e3:.2f} mm, {nf} fluid + {ns} solid → {gsd_file}')
