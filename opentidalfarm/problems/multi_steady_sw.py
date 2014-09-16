from problem import Problem
from steady_sw import SteadySWProblemParameters
from steady_sw import SteadySWProblem
from dolfin_adjoint import Constant


class MultiSteadySWProblemParameters(SteadySWProblemParameters):
    """ A set of parameters for a :class:`MultiSteadySWProblem`.

    The parameters are as described in
    :class:`opentidalfarm.problems.steady_sw.SteadySWProblemParameters`.

    In addition following parameters are available:

    Temporal parameters:

    :ivar dt: The timestep. Default: 1.0.
    :ivar start_time: The start time. Default: 0.0.
    :ivar finish_time: The finish time. Default: 100.0.
    """

    # Time parameters
    dt = 1.
    start_time = 0.0
    finish_time = 100.0

    # Functional time integration parameters
    functional_final_time_only = False

class MultiSteadySWProblem(SteadySWProblem):
    r""" Create a shallow water problem consisting of a sequence of
    (independent) steady-state shallow water problems. More specifically, it
    solves for each time-level :math:`n`:

        .. math:: -\nabla\cdot\nu\nabla u^n+u^n\cdot\nabla u^n+g\nabla
            \eta^n + \frac{c_b + c_t}{H^n} \| u^n\| u^n &= f_u^n, \\
            \nabla \cdot \left( H^n u^n \right) &= 0,

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

        :parameter parameters: A :class:`MultiSteadySWProblemParameters`
            object containing the parameters of the problem.
    """

    def __init__(self, parameters):

        if not type(parameters) == MultiSteadySWProblemParameters:
            raise TypeError("parameters must be of type \
MultiSteadySWProblemParameters.")

        if float(parameters.start_time) >= float(parameters.finish_time):
            raise ValueError("start_time must be < finish_time.")

        super(MultiSteadySWProblem, \
                self).__init_without_type_check__(parameters)

    @property
    def _is_transient(self):
        return True

    @staticmethod
    def default_parameters():
        ''' Returns a dictionary with the default parameters '''

        return MultiSteadySWProblemParameters()
