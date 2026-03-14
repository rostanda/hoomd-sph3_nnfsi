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

Creates the initial GSD file for the two-phase Power-Law Couette benchmark (03).

Written file:
  couette_{num_length}_init.gsd — two-fluid Couette, top wall at U_wall=0.01 m/s
    Types: W (lower Power-Law fluid), N (upper Newtonian fluid), S (solid walls)

Usage:
    python3 create_domain.py [num_length]
    Default: num_length=20
"""

import sys, os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, os.path.join(_ROOT, 'helper_modules'))

import hoomd
from hoomd import sph
import numpy as np
import gsd.hoomd

num_length = int(sys.argv[1]) if len(sys.argv) > 1 else 20

lref     = 0.001
H        = lref
U_wall   = 0.01    # top wall velocity [m/s]
rho0     = 1000.0
dx       = lref / num_length
n_solid  = 3
mass     = rho0 * dx**3

kernel   = 'WendlandC4'
slength  = hoomd.sph.kernel.OptimalH[kernel] * dx
rcut     = hoomd.sph.kernel.Kappa[kernel] * slength


def make_couette_gsd(filename):
    """Two-fluid Couette: W layer (lower), N layer (upper), S walls."""
    nx = num_length; ny = num_length + 2*n_solid; nz = num_length
    lx, ly, lz = nx*dx, ny*dx, nz*dx
    y_bot, y_top = -(H/2 + n_solid*dx), H/2 + n_solid*dx
    x_arr = np.linspace(-lx/2+dx/2, lx/2-dx/2, nx)
    y_arr = np.linspace(y_bot+dx/2,  y_top-dx/2, ny)
    z_arr = np.linspace(-lz/2+dx/2,  lz/2-dx/2, nz)
    xg, yg, zg = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
    pos = np.stack((xg.ravel(), yg.ravel(), zg.ravel()), axis=1).astype(np.float32)
    N   = len(pos); y = pos[:, 1]
    tid = np.zeros(N, dtype=np.int32)
    vel = np.zeros((N, 3), dtype=np.float32)
    tid[(y >= 0.0) & (y < H/2)] = 1    # 'N' upper fluid
    sol_bot = y < -H/2;  sol_top = y >= H/2
    tid[sol_bot | sol_top] = 2          # 'S' solid walls
    vel[sol_top, 0] = U_wall
    snap = gsd.hoomd.Frame()
    snap.configuration.box  = [lx, ly, lz, 0, 0, 0]
    snap.particles.N        = N
    snap.particles.types    = ['W', 'N', 'S']
    snap.particles.typeid   = tid
    snap.particles.position = pos
    snap.particles.velocity = vel
    snap.particles.mass     = np.full(N, mass,    dtype=np.float32)
    snap.particles.slength  = np.full(N, slength, dtype=np.float32)
    snap.particles.density  = np.full(N, rho0,    dtype=np.float32)
    with gsd.hoomd.open(filename, 'w') as f:
        f.append(snap)
    return int(np.sum(tid==0)), int(np.sum(tid==1)), int(np.sum(tid==2))


gsd_file = f'couette_{num_length}_init.gsd'
nW, nN, nS = make_couette_gsd(gsd_file)
print(f'Two-phase Couette geometry (num_length={num_length}): {nW} W + {nN} N + {nS} S → {gsd_file}')
