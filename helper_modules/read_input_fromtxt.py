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

import hoomd
from hoomd import *
from hoomd import sph
import numpy as np
import itertools
import gsd.hoomd
import os


def sanity_check_input(input_dict):
    """Validate the input parameter dictionary read from a simulation input file.

    Raises ``ValueError`` if any parameter is out of range or inconsistent.
    Called automatically by :func:`get_input_data_from_file`.

    Parameters
    ----------
    input_dict : dict
        Parameter dictionary as returned by :func:`get_input_data_from_file`.
        Must contain the keys: ``rawfilename``, ``kernel``, ``nx``, ``ny``,
        ``nz``, ``fdensity``, ``fviscosity``, ``delete_flag``, ``porosity``.

    Returns
    -------
    None
    """

    fname = input_dict['rawfilename']

    # check if file exists
    if os.path.exists(fname) == False:
        raise ValueError(f'File {fname} does not exist!')

    # check kernel
    list_of_kernels = ['WendlandC2', 'WendlandC4', 'WendlandC6', 'Quintic', 'CubicSpline']
    kernel = input_dict['kernel']
    if kernel not in list_of_kernels:
        raise ValueError(f'Kernel {kernel} not a member of available kernels.')

    # check domain sizes
    nx = input_dict['nx']
    ny = input_dict['ny']
    nz = input_dict['nz']

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError('Domain sizes can not be <= 0.')

    # check fluid parameters
    fdensity = input_dict['fdensity']
    if fdensity <= 0:
        raise ValueError('Density has to be > 0.')

    fviscosity = input_dict['fviscosity']
    if fviscosity <= 0:
        raise ValueError('Viscosity has to be > 0.')

    d_handle = input_dict['delete_flag']
    if d_handle != 0 and d_handle != 1:
        raise ValueError('Flag on deleting solids not set properly.')

    porosity = input_dict['porosity']
    if porosity < 0.0 or porosity > 1.0:
        raise ValueError('Porosity has to be 0 <= phi <= 1.')



def get_input_data_from_file(inputfile, TEST_SANITY = True):
    """Read simulation input parameters from a structured text file.

    The text file must follow the layout below (two header lines are skipped,
    then string parameters, then float parameters; the last 7 lines are
    skipped by the float parser)::

        # SPH simulation input file
        #
        path/to/raw_geometry_file.raw
        WendlandC2
        0.001       # vsize  – voxel/particle spacing [m]
        100         # nx     – number of voxels in x
        50          # ny     – number of voxels in y
        50          # nz     – number of voxels in z
        1000.0      # fdensity   – fluid density [kg/m³]
        0.001       # fviscosity – dynamic viscosity [Pa·s]
        1           # delete_flag – 1: delete interior solids, 0: keep all
        0.4         # porosity – volume fraction of void space [0, 1]

    Parameters
    ----------
    inputfile : str or path-like
        Path to the ``.txt`` input file.

    Returns
    -------
    dict
        Dictionary with keys: ``rawfilename`` (str), ``kernel`` (str),
        ``vsize`` (float64), ``nx`` (int32), ``ny`` (int32), ``nz`` (int32),
        ``fdensity`` (float64), ``fviscosity`` (float64),
        ``delete_flag`` (int32), ``porosity`` (float64).

    Raises
    ------
    ValueError
        If any parameter fails the sanity check (see
        :func:`sanity_check_input`).
    """
    parameter_dict = {}

    strs = np.genfromtxt(inputfile, str, skip_header = 2, skip_footer = 7, usecols = (0, ))
    flts = np.genfromtxt(inputfile, np.float64, skip_header = 4, usecols = (0, ))


    parameter_dict.update({'rawfilename' : str(strs[0]) })
    parameter_dict.update({'kernel'      : str(strs[1]) })

    parameter_dict.update({'vsize'       : np.float64(flts[0]) })
    parameter_dict.update({'nx'          : np.int32(flts[1]) })
    parameter_dict.update({'ny'          : np.int32(flts[2]) })
    parameter_dict.update({'nz'          : np.int32(flts[3]) })
    parameter_dict.update({'fdensity'    : np.float64(flts[4]) })
    parameter_dict.update({'fviscosity'  : np.float64(flts[5]) })
    parameter_dict.update({'delete_flag' : np.int32(flts[6]) })
    parameter_dict.update({'porosity'    : np.float64(flts[7]) })

    if TEST_SANITY:
        sanity_check_input(parameter_dict)

    return parameter_dict
