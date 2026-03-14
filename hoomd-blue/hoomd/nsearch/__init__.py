# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Molecular dynamics.

In molecular dynamics simulations, HOOMD-blue numerically integrates the degrees
of freedom in the system as a function of time under the influence of forces. To
perform MD simulations, assign a MD `Integrator` to the `hoomd.Simulation`
operations. Provide the `Integrator` with lists of integration methods, forces,
and constraints to apply during the integration. Use `hoomd.md.minimize.FIRE`
to perform energy minimization.

MD updaters (`hoomd.md.update`) perform additional operations during the
simulation, including rotational diffusion and establishing shear flow.
Use MD computes (`hoomd.md.compute`) to compute the thermodynamic properties of
the system state.

See Also:
    Tutorial: :doc:`tutorial/01-Introducing-Molecular-Dynamics/00-index`
"""

from hoomd.nsearch import nlist
from hoomd.nsearch.half_step_hook import HalfStepHook

__all__ = [
    "nlist",
    "HalfStepHook",
]