#!/usr/bin/env python3

"""
Copyright (c) 2025-2026 David Krach, Daniel Rostan.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

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
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

maintainer: dkrach, david.krach@mib.uni-stuttgart.de

"""

# --- HEADER -------------------------------------------------
import hoomd
from hoomd import *
from hoomd import sph
import numpy as np
import itertools
import gsd.hoomd
# ------------------------------------------------------------


def print_decomp_info(sim, device):
    """Print MPI domain-decomposition information to stdout.

    Outputs the domain decomposition layout, the positions of the split
    planes, and the number of particles held by each MPI rank.  Only the
    domain decomposition and split-plane lines are printed on rank 0; the
    per-rank particle count is printed by every rank.

    Parameters
    ----------
    sim : hoomd.Simulation
        Active HOOMD simulation object.
    device : hoomd.device.Device
        HOOMD device; used to identify rank 0 for selective output.

    Returns
    -------
    None
    """

    # Print the domain decomposition.
    domain_decomposition = sim.state.domain_decomposition
    if device.communicator.rank == 0:
        print(f'Domain Decomposition: {domain_decomposition}')

    # Print the location of the split planes.
    split_fractions = sim.state.domain_decomposition_split_fractions
    if device.communicator.rank == 0:
        print(f'Locations of SplitPlanes: {split_fractions}')

    # Print the number of particles on each rank.
    with sim.state.cpu_local_snapshot as snap:
        N = len(snap.particles.position)
        print(f'{N} particles on rank {device.communicator.rank}')