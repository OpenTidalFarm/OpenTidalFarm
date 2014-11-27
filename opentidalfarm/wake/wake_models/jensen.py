"""
Implements the Jensen wake model.
"""
from base_wake import WakeModel
from ...helpers import FrozenClass
import numpy


class JensenParameters(FrozenClass):
    """Defins a set of parameters for the Jensen Model."""
    thrust_coefficient = 0.6
    wake_decay = 0.03
    turbine_radius = None


class Jensen(WakeModel):
    """Implements a Jensen wake model."""
    def __init__(self, parameters, flow_field):
        if not isinstance(parameters, JensenParameters):
            raise TypeError("'parameters' must be of type 'JensenParameters'")

        if parameters.turbine_radius is None:
            raise ValueError("'parameters.turbine_radius' cannot be None")

        self.parameters = parameters

        # Call the parent constructor.
        super(Jensen, self).__init__(flow_field)

    @staticmethod
    def default_parameters():
        return JensenParameters()

    def _wake_radius(self, distance):
        """Returns the radius of the wake a 'distance' downstream."""
        turbine_radius = self.parameters.turbine_radius
        wake_decay = self.parameters.wake_decay
        return (turbine_radius*(1. + 2*wake_decay* (distance/2*turbine_radius)))

    def in_wake(self, relative_position):
        """True if turbine is in the wake of another_turbine."""
        x0, y0 = relative_position
        if (x0 < 0.):
            return False
        else:
            wake_radius = self._wake_radius(x0)
            return wake_radius > abs(y0)

    def multiplier(self, turbine, another_turbine):
        """Returns the a multiplier to scale the ambient flow by."""
        relative_position = self.relative_position(turbine, another_turbine)
        if self.in_wake(relative_position):
            thrust = self.parameters.thrust_coefficient
            wake_decay = self.parameters.wake_decay
            radius = self.parameters.turbine_radius
            x0, y0 = relative_position
            return 1.0 - (1.0 - thrust)**0.5/((1.0 + wake_decay*x0/radius)**2)
        else:
            return 1.0
