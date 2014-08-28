from problem import Problem
from steady_shallow_water import SteadyShallowWaterProblemParameters
from steady_shallow_water import SteadyShallowWaterProblem
from dolfin_adjoint import Constant


class MultiSteadyShallowWaterProblemParameters(SteadyShallowWaterProblemParameters):
    """ A set of parameters for a :class:`MultiSteadyShallowWaterProblem`.

    The parameters are described as in
    :class:`SteadyShallowWaterProblemParameters`.

    In addition following parameters are available:

    Domain parameters:

    :ivar domain: The computational domain as an :class:`Domain` object.

    Time parameters:

    :ivar dt: The timestep. Default: 1.0.
    :ivar start_time: The start time. Default: 0.0.
    :ivar finish_time: The finish time. Default: 100.0.
    """

    # Time parameters
    dt = 1.
    start_time = 0.0
    finish_time = 100.0


class ShallowWaterProblem(SteadyShallowWaterProblem):

    def __init__(self, parameters):
        """ Instantiates a new :class:`ShallowWaterProblem` object. 

            :parameter parameters: A :class:`ShallowWaterProblemParameters`
                object containing the parameters of the problem.
        """

        super(ShallowWaterProblem, self).__init__(parameters)

    @property
    def _is_transient(self):
        return True

    @staticmethod
    def default_parameters():
        ''' Returns a dictionary with the default parameters '''

        return ShallowWaterProblemParameters()
