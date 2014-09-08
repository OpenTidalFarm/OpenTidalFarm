from dolfin_adjoint import Constant
from problem import Problem
from ..helpers import FrozenClass
from .. import finite_elements


class DummyProblemParameters(FrozenClass):
    """ A set of parameters for a :class:`DummyProblem`.
    """

    domain = None
    dt = None
    rho = 1000.

    # Finite element settings
    finite_element = staticmethod(finite_elements.p2p1)

    # Initial condition
    initial_condition = Constant((1e-16, 0, 0))

    # Tidal farm
    tidal_farm = None

    functional_final_time_only = False


class DummyProblem(Problem):

    def __init__(self, parameters):
        """ Instantiates a new :class:`DummyProblem` object.

            :parameter parameters: A :class:`DummyProblemParameters`
                object containing the parameters of the problem.
        """

        if not isinstance(parameters, DummyProblemParameters):
            raise TypeError("parameters must be of type \
DummyProblemParameters.")

        self.parameters = parameters

    @property
    def _is_transient(self):
        return True

    @staticmethod
    def default_parameters():
        ''' Returns a :class:`DummyProblemParameters` with default
            parameters. '''

        return DummyProblemParameters()
