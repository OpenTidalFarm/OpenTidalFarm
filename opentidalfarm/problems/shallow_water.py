from problem import Problem
from steady_shallow_water import SteadyShallowWaterProblemParameters
from steady_shallow_water import SteadyShallowWaterProblem
from dolfin_adjoint import Constant


class ShallowWaterProblemParameters(SteadyShallowWaterProblemParameters):
    """ A set of parameters for a :class:`ShallowWaterProblem`.

    The parameters are described as in
    :class:`SteadyShallowWaterProblemParameters`.

    In addition following parameters are available:

    Domain parameters:

    :ivar domain: The computational domain as an :class:`Domain` object.

    Time parameters:

    :ivar theta: The theta value for the timestepping-scheme. Default 1.0.
    :ivar dt: The timestep. Default: 1.0.
    :ivar start_time: The start time. Default: 0.0.
    :ivar finish_time: The finish time. Default: 100.0.

    Equation parameters:

    :ivar include_time_term: Boolean indicating if the time term is included.
                             Default: True

    Functional time integration paramters (FIXME: Move to reduced functional):

    :ivar functional_final_time_only: Boolean indicating if the functional
        should be integrated over time or evaluated at the end of time only. 
        Default: True.
    """

    # Time parameters
    theta = 1.0
    dt = 1.
    start_time = 0.0
    finish_time = 100.0

    # Equation settings
    include_time_term = True

    # Functional time integration parameters
    functional_final_time_only = True


class ShallowWaterProblem(SteadyShallowWaterProblem):

    def __init__(self, parameters, check_parameter_type=True):
        """ Instantiates a new :class:`ShallowWaterProblem` object. 

            :parameter parameters: A :class:`ShallowWaterProblemParameters`
                object containing the parameters of the problem.
        """

        if (check_parameter_type and
            not type(parameters) == ShallowWaterProblemParameters):
            raise TypeError("parameters must be of type \
ShallowWaterProblemParameters.")

        if float(parameters.start_time) >= float(parameters.finish_time):
            raise ValueError("start_time must be < finish_time.")

        super(ShallowWaterProblem, self).__init__(parameters, False)

    @property
    def _is_transient(self):
        return True

    @staticmethod
    def default_parameters():
        ''' Returns a dictionary with the default parameters '''

        return ShallowWaterProblemParameters()
