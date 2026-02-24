# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement SPH  Integrator."""

import hoomd
import hoomd.sph
from hoomd import _hoomd
from hoomd.sph import _sph
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.operation import Integrator as BaseIntegrator
from hoomd.data import syncedlist
from hoomd.sph.methods import Method
from hoomd.sph.force import Force
from hoomd.sph.constrain import Constraint


def _set_synced_list(old_list, new_list):
    old_list.clear()
    old_list.extend(new_list)


class _DynamicIntegrator(BaseIntegrator):

    def __init__(self, forces, constraints, methods):
        forces = [] if forces is None else forces
        constraints = [] if constraints is None else constraints
        methods = [] if methods is None else methods
        self._forces = syncedlist.SyncedList(
            Force, syncedlist._PartialGetAttr("_cpp_obj"), iterable=forces)

        self._constraints = syncedlist.SyncedList(
            OnlyTypes(Constraint),
            syncedlist._PartialGetAttr("_cpp_obj"),
            iterable=constraints)

        self._methods = syncedlist.SyncedList(
            Method, syncedlist._PartialGetAttr("_cpp_obj"), iterable=methods)

    def _attach_hook(self):
        self._forces._sync(self._simulation, self._cpp_obj.forces)
        self._constraints._sync(self._simulation, self._cpp_obj.constraints)
        self._methods._sync(self._simulation, self._cpp_obj.methods)
        super()._attach_hook()

    def _post_attach_hook(self):
        self.validate_groups()

    def _detach_hook(self):
        self._forces._unsync()
        self._methods._unsync()
        self._constraints._unsync()
        super()._detach()

    def validate_groups(self):
        """Verify groups.

        Groups may change after attaching.
        Users can call `validate_groups` to verify the groups after changing
        them.
        """
        self._cpp_obj.validate_groups()

    @property
    def forces(self):
        return self._forces

    @forces.setter
    def forces(self, value):
        _set_synced_list(self._forces, value)

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        _set_synced_list(self._constraints, value)

    @property
    def methods(self):
        return self._methods

    @methods.setter
    def methods(self, value):
        _set_synced_list(self._methods, value)

    def _setattr_param(self, attr, value):
        super()._setattr_param(attr, value)


@hoomd.logging.modify_namespace(("sph", "Integrator"))
class Integrator(_DynamicIntegrator):
    r"""SPH time integration.

    Args:
        dt (float): Integrator time step size :math:`[\mathrm{time}]`.

        methods (Sequence[hoomd.sph.methods.Method]): Sequence of integration
          methods (e.g. `VelocityVerlet`, `KickDriftKickTV`). The default value
          of ``None`` initializes an empty list.

        forces (Sequence[hoomd.sph.force.Force]): Sequence of forces applied to
          the particles in the system. The default value of ``None`` initializes
          an empty list.

        constraints (Sequence[hoomd.sph.constrain.Constraint]): Sequence of
          constraint forces. The default value of ``None`` initializes an empty
          list.

        half_step_hook (hoomd.sph.HalfStepHook): Optional callback executed
            during the half-step of the integration.

    `Integrator` orchestrates the SPH time integration step. Each method in
    `methods` applies its equations of motion to a subset of particles. The
    particle subsets must not overlap.

    Attributes:
        dt (float): Integrator time step size :math:`[\mathrm{time}]`.

        methods (list[hoomd.sph.methods.Method]): List of integration methods.

        forces (list[hoomd.sph.force.Force]): List of forces applied to
            the particles in the system.

        constraints (list[hoomd.sph.constrain.Constraint]): List of
            constraint forces applied to the particles in the system.

        half_step_hook (hoomd.sph.HalfStepHook): User defined half-step
            callback.

    Examples::

        nlist = hoomd.nsearch.nlist.Tree()
        kernel = hoomd.sph.kernel.WendlandC2()
        eos = hoomd.sph.eos.Tait()
        spf = hoomd.sph.sphmodel.SinglePhaseFlow(kernel=kernel, eos=eos,
                                                  nlist=nlist, ...)
        method = hoomd.sph.methods.VelocityVerlet(filter=hoomd.filter.All(),
                                                   densitymethod='SUMMATION')
        integrator = hoomd.sph.Integrator(dt=1e-4, methods=[method],
                                           forces=[spf])
        sim.operations.integrator = integrator

    """

    def __init__(self,
                 dt,
                 forces=None,
                 constraints=None,
                 methods=None,
                 half_step_hook=None
                 ):

        super().__init__(forces, constraints, methods)

        self._param_dict.update(
            ParameterDict(
                dt=float(dt),
                half_step_hook=OnlyTypes(hoomd.sph.HalfStepHook,
                                         allow_none=True)))

        self.half_step_hook = half_step_hook

    def _attach_hook(self):
        # initialize the reflected c++ class
        self._cpp_obj = _sph.SPHIntegratorTwoStep(
            self._simulation.state._cpp_sys_def, self.dt)

        # Call attach from DynamicIntegrator which attaches forces,
        # constraint_forces, and methods, and calls super()._attach() itself.
        super()._attach_hook()

    @hoomd.logging.log(category="sequence", requires_run=True)
    def linear_momentum(self):
        """tuple(float,float,float): The linear momentum vector of the system \
            :math:`[\\mathrm{mass} \\cdot \\mathrm{velocity}]`.

        .. math::

            \\vec{p} = \\sum_{i=0}^\\mathrm{N_particles-1} m_i \\vec{v}_i
        """
        v = self._cpp_obj.computeLinearMomentum()
        return (v.x, v.y, v.z)
