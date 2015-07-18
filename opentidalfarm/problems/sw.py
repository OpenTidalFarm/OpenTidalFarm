from problem import Problem
from steady_sw import SteadySWProblemParameters
from steady_sw import SteadySWProblem
from dolfin_adjoint import Constant


class SWProblemParameters(SteadySWProblemParameters):
    """ A set of parameters for a :class:`SWProblem`.

    The parameters are as described in
    :class:`opentidalfarm.problems.steady_sw.SteadySWProblemParameters`.

    In addition following parameters are available:

    Time parameters:

    :ivar theta: The theta value for the timestepping-scheme. Default 1.0.
    :ivar dt: The timestep. Default: 1.0.
    :ivar start_time: The start time. Default: 0.0.
    :ivar finish_time: The finish time. Default: 100.0.

    Functional time integration paramters (FIXME: Move to reduced functional):

    :ivar functional_final_time_only: Boolean indicating if the functional
        should be integrated over time or evaluated at the end of time only.
        Default: False.
    """

    # Time parameters
    theta = 1.0
    dt = 1.
    start_time = 0.0
    finish_time = 100.0

    # Functional time integration parameters
    functional_final_time_only = False


class SWProblem(SteadySWProblem):
    r""" Create a transient shallow water problem:

        .. math:: \frac{\partial u}{\partial t} -\nabla\cdot\nu\nabla u+u\cdot\nabla u+g\nabla
            \eta + \frac{c_b + c_t}{H} \| u\| u &= f_u, \\
            \frac{\partial \eta}{\partial t} + \nabla \cdot \left( H u \right) &= 0,

        where

        - :math:`u` is the velocity,
        - :math:`\eta` is the free-surface displacement,
        - :math:`H=\eta + h` is the total water depth where :math:`h` is the
          water depth at rest,
        - :math:`f_u` is the velocity forcing term,
        - :math:`c_b` is the (quadratic) natural bottom friction coefficient,
        - :math:`c_t` is the (quadratic) friction coefficient due to the turbine
          farm,
        - :math:`\nu` is the viscosity coefficient,
        - :math:`g` is the gravitational constant,


        :parameter parameters: A :class:`SWProblemParameters`
            object containing the parameters of the problem.
    """

    def __init__(self, parameters):

        if not type(parameters) == SWProblemParameters:
            raise TypeError("parameters must be of type SWProblemParameters.")

        if float(parameters.start_time) >= float(parameters.finish_time):
            raise ValueError("start_time must be < finish_time.")

        super(SWProblem, self).__init_without_type_check__(parameters)

    @property
    def _is_transient(self):
        return True

    @staticmethod
    def default_parameters():
        ''' Returns a dictionary with the default parameters '''

        return SWProblemParameters()
