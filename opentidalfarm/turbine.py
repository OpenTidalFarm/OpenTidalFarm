import dolfin
import copy
from table import tabulate_data

class Turbine(object):
    """Turbine parametrs."""
    def __init__(self, diameter, minimum_distance, friction, coordinates=None):
        self._diameter = diameter
        self._minimum_distance = minimum_distance
        self._friction = friction
        self._coordinates = coordinates


    def __str__(self):
        """String representation of turbine properties."""
        # Properly represent the coordinates.
        if self._coordinates is not None:
            coordinates = "(%.2f, %.2f)" % (self.coordinates)
        else:
            coordinates = "not yet positioned"

        header = "Turbine specification"
        data = [("Diameter / m", self.diameter),
                ("Minimum distance / m", self.minimum_distance),
                ("Friction coefficient", self.friction),
                ("Coordinates / (m, m)", coordinates)]
        return tabulate_data(header, data)


    def _copy_constructor(self, coordinates=None):
        """Creates a copy of the current turbine and places it at coordinates.

        Args:
            coordinates: A tuple of the x-y coordinates defining the location of
                the new turbine.
        """
        turbine = copy.deepcopy(self)
        turbine.coordinates = coordinates
        return turbine

    @property
    def friction(self):
        """The friction coefficient of a turbine"""
        return self._friction


    @friction.setter
    def friction(self, value):
        self._friction = value

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
    def coordinates(self):
        """The coordinates of the turbine"""
        if self._coordinates is not None:
            return self._coordinates
        else:
            raise RuntimeError("This turbine has not been positioned.")


    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = value


    @property
    def x(self):
        """Return the x-coordinate of the turbine"""
        return self.coordinates[0]


    @property
    def y(self):
        """Return the y-coordinate of the turbine"""
        return self.coordinates[1]
