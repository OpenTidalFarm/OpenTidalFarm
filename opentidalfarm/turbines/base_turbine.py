import dolfin

class BaseTurbine(object):
    """A base turbine class from which others are derived."""
    def __init__(self, friction=None, diameter=None, minimum_distance=None,
                 controls=None, bump=False, smeared=False, thrust=False,
                 implicit_thrust=False):
        # Possible turbine parameters.
        self._diameter = diameter
        self._minimum_distance = minimum_distance
        self._friction = friction
        self._controls = controls

        # Possible parameterisations.
        self._bump = bump
        self._smeared = smeared
        self._thrust = thrust
        self._implicit_thrust = implicit_thrust

        # The integral of the unit bump function computed with Wolfram Alpha:
        # "integrate e^(-1/(1-x**2)-1/(1-y**2)+2) dx dy,
        #  x=-0.999..0.999, y=-0.999..0.999"
        # http://www.wolframalpha.com/input/?i=integrate+e%5E%28-1%2F%281-x**2%29-1%2F%281-y**2%29%2B2%29+dx+dy%2C+x%3D-0.999..0.999%2C+y%3D-0.999..0.999
        self._unit_bump_int = 1.45661


    @property
    def friction(self):
        """The maximum friction coefficient of a turbine.
        :returns: The maximum friction coefficient of the turbine.
        :rtype: float
        """
        if self._friction is None:
            raise ValueError("Friction has not been set!")
        return self._friction


    @property
    def diameter(self):
        """The diameter of a turbine.
        :returns: The diameter of a turbine.
        :rtype: float
        """
        if self._diameter is None:
            raise ValueError("Diameter has not been set!")
        return self._diameter


    @property
    def radius(self):
        """The radius of a turbine.
        :returns: The radius of a turbine.
        :rtype: float
        """
        return self.diameter*0.5


    @property
    def minimum_distance(self):
        """The minimum distance allowed between turbines.
        :returns: The minimum distance allowed between turbines.
        :rtype: float
        """
        if self._minimum_distance is None:
            raise ValueError("Minimum distance has not been set!")
        return self._minimum_distance


    @property
    def integral(self):
        """The integral of the turbine bump function.
        :returns: The integral of the turbine bump function.
        :rtype: float
        """
        return self._unit_bump_int*self._diameter/4.


    def _set_controls(self, controls):
        self._controls = controls

    def _get_controls(self):
        if self._controls is not None:
            return self._controls
        else:
            raise ValueError("The controls have not been set.")

    controls = property(_get_controls, _set_controls, "The turbine controls.")


    @property
    def bump(self):
        return self._bump

    @property
    def smeared(self):
        return self._smeared

    @property
    def thrust(self):
        return self._thrust

    @property
    def implicit_thrust(self):
        return self._implicit_thrust
