from problem import Problem
from steady_shallow_water import SteadyShallowWaterProblemParameters


class ShallowWaterProblemParameters(SteadyShallowWaterProblemParameters):
    """ A set of parameters for a :class:`SteadyShallowWaterProblem`.

    The parameters are described as in
    :class:`SteadyShallowWaterProblemParameters`.

    In addition following parameters are available:

    Time parameters:

    :ivar theta: The theta value for the timestepping-scheme. Default 1.0.
    :ivar dt: The timestep. Default: 1.0.
    :ivar start_time: The start time. Default: 0.0.
    :ivar current_time: The current simulation time. Default: 0.0.
    :ivar finish_time: The finish time. Default: 100.0.
    :ivar t: FIXME: Remove.

    Equation parameters:

    :ivar include_time_term: Boolean indicating if the time term is included.
                             Default: True

    Functional time integration paramters (FIXME: Move to reduced functional):

    :ivar functional_final_time_only: Boolean indicating if the functional
        should be integrated over time or evaluated at the end of time only. 
        Default: True.
    :ivar functional_quadrature_degree: The quadrature degree of the functional
        integration. Used only if :attr:`functional_final_time_only:` is True. 
        Default: 1.
    """

    # Time parameters
    theta = 1.0
    dt = 1.
    start_time = 0.0
    current_time = 0.0
    finish_time = 100.0
    t = 0.0

    # Equation settings
    include_time_term = True

    # Functional time integration parameters
    functional_final_time_only = True
    functional_quadrature_degree = 1


class ShallowWaterProblem(Problem):

    def __init__(self, parameters):
        """ Instantiates a new :class:`ShallowWaterProblem` object. 

            :parameter parameters: A :class:`ShallowWaterProblemParameters`
                object containing the parameters of the problem.
        """

        if not isinstance(parameters, ShallowWaterProblemParameters):
            raise TypeError, "parameters must be of type \
ShallowWaterProblemParameters."

        self.parameters = parameters

    @property
    def _is_transient(self):
        return True

    @staticmethod
    def default_parameters():
        ''' Returns a dictionary with the default parameters '''

        return ShallowWaterProblemParameters()
