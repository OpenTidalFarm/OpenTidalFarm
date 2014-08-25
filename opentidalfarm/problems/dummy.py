from problem import Problem
from ..helpers import FrozenClass


class DummyProblemParameters(FrozenClass):
    """ A set of parameters for a :class:`DummyProblem`.
    """

    pass


class DummyProblem(Problem):

    def __init__(self, parameters):
        """ Instantiates a new :class:`DummyProblem` object.

            :parameter parameters: A :class:`DummyProblemParameters`
                object containing the parameters of the problem.
        """

        if not isinstance(parameters, DummyProblem):
            raise TypeError("parameters must be of type \
ShallowWaterProblemParameters.")

        self.parameters = parameters

    @property
    def _is_transient(self):
        return False

    @staticmethod
    def default_parameters():
        ''' Returns a :class:`DummyProblemParameters` with default
            parameters. '''

        return DummyProblemParameters()
