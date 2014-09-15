import dolfin
import copy
from ..controls import Controls

class Turbine(object):
    """Turbine parameters."""
    def __init__(self, diameter, minimum_distance, maximum_friction,
                 controls=None):
        self._diameter = diameter
        self._minimum_distance = minimum_distance
        self._maximum_friction = maximum_friction
        self._controls = controls


    @property
    def friction(self):
        """The maximum friction coefficient of a turbine"""
        return self._maximum_friction


    @property
    def diameter(self):
        """The diameter of a turbine"""
        return self._diameter


    @property
    def radius(self):
        """The radius of a turbine"""
        return self._diameter*0.5


    @property
    def minimum_distance(self):
        """The minimum distance between turbines"""
        return self._minimum_distance


    def _set_controls(self, controls):
        self._controls = controls


    def _get_controls(self):
        if self._controls is not None:
            return self._controls
        else:
            raise ValueError("The controls have not been set!")


    controls = property(_get_controls, _set_controls, "The turbine controls.")
