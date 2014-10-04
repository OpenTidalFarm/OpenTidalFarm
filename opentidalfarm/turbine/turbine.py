import dolfin
import copy
from ..controls import Controls

class Turbine(object):
    """Turbine parameters."""
    def __init__(self, diameter, minimum_distance, maximum_friction,
                 controls=None, parameterisation=None):
        self._diameter = diameter
        self._minimum_distance = minimum_distance
        self._maximum_friction = maximum_friction
        self._controls = controls
        self._parameterisation = parameterisation

        # The integral of the unit bump function computed with Wolfram Alpha:
        # "integrate e^(-1/(1-x**2)-1/(1-y**2)+2) dx dy,
        #  x=-0.999..0.999, y=-0.999..0.999"
        # http://www.wolframalpha.com/input/?i=integrate+e%5E%28-1%2F%281-x**2%29-1%2F%281-y**2%29%2B2%29+dx+dy%2C+x%3D-0.999..0.999%2C+y%3D-0.999..0.999
        self._unit_bump_int = 1.45661

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


    @property
    def integral(self):
        """The integral of the turbine."""
        return self._unit_bump_int*self._diameter/4.


    def _set_controls(self, controls):
        self._controls = controls

    def _get_controls(self):
        if self._controls is not None:
            return self._controls
        else:
            raise ValueError("The controls have not been set.")

    controls = property(_get_controls, _set_controls, "The turbine controls.")


    def _set_parameterisation(self, parameterisation):
        self._parameterisation = parameterisation

    def _get_parameterisation(self):
        if self._parameterisation is not None:
            return self._parameterisation
        else:
            raise ValueError("The turbines have not yet been parameterised.")

    parameterisation = property(_get_parameterisation, _set_parameterisation,
                                "The turbine parameterisation.")
