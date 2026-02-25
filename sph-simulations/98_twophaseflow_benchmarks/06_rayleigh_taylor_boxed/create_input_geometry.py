
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

Rayleigh–Taylor instability (boxed) — initial geometry creation.

BENCHMARK DESCRIPTION
---------------------
A heavy fluid (phase 'W', ρ₁ = 1500 kg/m³) sits on top of a light fluid
(phase 'N', ρ₂ = 500 kg/m³) in a rectangular domain.  Gravity acts downward
(−y).  The interface is perturbed sinusoidally in x:

    y_interface(x) = δ × cos(2π x / lref)

with amplitude δ = 0.01 × lref.  This seeds the dominant mode and the
interface subsequently develops into a single RT plume.

Domain: lref × 4·lref × lref  (solid walls in x and y; periodic in z).
Atwood number: At = (ρ₁ − ρ₂) / (ρ₁ + ρ₂) = 0.5

Linear growth rate (inviscid):
    γ = sqrt(At × g × k) = sqrt(0.5 × 9.81 × 2π / lref) ≈ 176 s⁻¹

The nonlinear bubble/spike evolution is compared against the periodic-x
variant (05_rayleigh_taylor) to study confinement effects.

Usage:
    python3 create_input_geometry.py <num_length>

    num_length : integer resolution — particles across lref (e.g. 20, 40)

Output:
    rt_boxed_<nx>_<ny>_<nz>_vs_<dx>_init.gsd
"""

import sys
import numpy as np
import gsd.hoomd
import hoomd
from hoomd import sph

# ─── Parameters ──────────────────────────────────────────────────────────────
num_length = int(sys.argv[1]) if len(sys.argv) > 1 else 20

lref    = 0.001                    # domain width = reference length    [m]
rho01   = 1500.0                   # rest density heavy fluid ('W')     [kg/m³]
rho02   = 500.0                    # rest density light fluid ('N')     [kg/m³]
dx      = lref / num_length        # particle spacing                   [m]
mass1   = rho01 * dx**3            # particle mass phase W              [kg]
mass2   = rho02 * dx**3            # particle mass phase N              [kg]
delta   = 0.01 * lref              # perturbation amplitude             [m]
n_solid = 3                        # solid wall layers on each side

kernel  = 'WendlandC4'
slength = hoomd.sph.kernel.OptimalH[kernel] * dx

# ─── Box geometry ────────────────────────────────────────────────────────────
# Fluid region: lref (x) × 4·lref (y) × lref (z).
# Solid walls on all four sides (left/right in x, top/bottom in y); z is periodic.
nx_flu = num_length
ny_flu = 4 * num_length
nz     = num_length

nx = nx_flu + 2 * n_solid
ny = ny_flu + 2 * n_solid

lx_flu = nx_flu * dx               # fluid width  = lref                [m]
H_flu  = ny_flu * dx               # fluid height = 4 × lref            [m]

lx = nx * dx
ly = ny * dx
lz = nz * dx

# ─── Particle lattice ────────────────────────────────────────────────────────
x_arr = np.linspace(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx, endpoint=True)
y_arr = np.linspace(-ly / 2 + dx / 2, ly / 2 - dx / 2, ny, endpoint=True)
z_arr = np.linspace(-lz / 2 + dx / 2, lz / 2 - dx / 2, nz, endpoint=True)

xg, yg, zg = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
positions = np.stack((xg.ravel(), yg.ravel(), zg.ravel()), axis=1).astype(np.float32)

n_particles = len(positions)
typeids    = np.zeros(n_particles, dtype=np.int32)    # default: 'W' (heavy, top)
velocities = np.zeros((n_particles, 3), dtype=np.float32)
masses     = np.full(n_particles, mass1, dtype=np.float32)
slengths   = np.full(n_particles, slength, dtype=np.float32)
densities  = np.full(n_particles, rho01, dtype=np.float32)

x_pos = positions[:, 0]
y_pos = positions[:, 1]

# Perturbed interface: y_int(x) = delta * cos(2π x / lref)
y_int = delta * np.cos(2.0 * np.pi * x_pos / lref)

# Light fluid ('N', index 1): y < y_int  (below the perturbed interface)
# Heavy fluid ('W', index 0): y > y_int  (above the perturbed interface)
below_interface = y_pos < y_int
typeids[below_interface]   = 1     # 'N' — light fluid
masses[below_interface]    = mass2
densities[below_interface] = rho02

# Solid walls: top/bottom in y AND left/right in x
solid_bot   = y_pos <  -H_flu / 2
solid_top   = y_pos >=  H_flu / 2
solid_left  = x_pos <  -lx_flu / 2
solid_right = x_pos >=  lx_flu / 2
is_solid    = solid_bot | solid_top | solid_left | solid_right
typeids[is_solid]   = 2
masses[is_solid]    = mass1
densities[is_solid] = rho01

# ─── GSD snapshot ────────────────────────────────────────────────────────────
snapshot = gsd.hoomd.Frame()
snapshot.configuration.box  = [lx, ly, lz, 0, 0, 0]
snapshot.particles.N         = n_particles
snapshot.particles.types     = ['W', 'N', 'S']
snapshot.particles.typeid    = typeids
snapshot.particles.position  = positions
snapshot.particles.velocity  = velocities
snapshot.particles.mass      = masses
snapshot.particles.slength   = slengths
snapshot.particles.density   = densities

init_filename = f'rt_boxed_{nx}_{ny}_{nz}_vs_{dx:.6f}_init.gsd'
with gsd.hoomd.open(name=init_filename, mode='w') as f:
    f.append(snapshot)

At     = (rho01 - rho02) / (rho01 + rho02)
g_mag  = 9.81
k_mode = 2 * np.pi / lref
gamma  = np.sqrt(At * g_mag * k_mode)

n_solid_x = int(np.sum(solid_left | solid_right))
n_solid_y = int(np.sum((solid_bot | solid_top) & ~(solid_left | solid_right)))

print(f'Created {init_filename}')
print(f'  Total particles  : {n_particles}')
print(f'  Heavy W (above)  : {int(np.sum(typeids==0))} particles, ρ={rho01} kg/m³')
print(f'  Light N (below)  : {int(np.sum(typeids==1))} particles, ρ={rho02} kg/m³')
print(f'  Solid S (y-walls): {n_solid_y} particles')
print(f'  Solid S (x-walls): {n_solid_x} particles')
print(f'  Atwood number    : At = {At:.3f}')
print(f'  Perturbation δ   : {delta*1e6:.1f} µm  ({delta/lref*100:.1f}% of lref)')
print(f'  Linear growth γ  : {gamma:.1f} s⁻¹  (inviscid, for g={g_mag} m/s²)')
