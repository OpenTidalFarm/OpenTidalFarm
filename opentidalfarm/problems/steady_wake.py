from problem import Problem
from ..wake import WakeModel
from ..helpers import FrozenClass
from ..domains.domain import Domain

class SteadyWakeProblemParameters(FrozenClass):
    """A parameters set for a :class:`SteadyWakeProblem`.

    Domain parameters:

    :ivar domain: The computational domain, see :doc:`opentidalfarm.domains`.
    """

    # Domain
    domain = None

    # Wake and combination models
    wake_model = None
    combination_model = None

    # The farm being optimised
    tidal_farm = None



class SteadyWakeProblem(Problem):
    """Create a steady-state wake problem."""
    def __init__(self, parameters):
        if not isinstance(parameters, SteadyWakeProblemParameters):
            raise TypeError("'parameters' must be of type "
                            "'SteadyWakeProblemParameters'")

        if not isinstance(parameters.domain, Domain):
            raise TypeError("'parameters.domain' is not a 'Domain' object")

        if not isinstance(parameters.wake_model, WakeModel):
            raise TypeError("'parameters.wake_model' is not a 'WakeModel' "
                            "object")

        self.parameters = parameters


    # TODO: is this necessary?
    @property
    def _is_transient(self):
        return False


    @staticmethod
    def default_parameters():
        """Returns a :class:`SteadyWakeProblemParameters` with default
        parameters.
        """
        return SteadyWakeProblemParameters()
