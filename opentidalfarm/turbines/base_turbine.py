import dolfin

class BaseTurbine(object):
    """A base turbine class from which others are derived."""
    def __init__(self, friction=None, diameter=None, minimum_distance=None,
            controls=None, smeared=False):
        # Possible turbine parameters.
        self._diameter = diameter
        self._minimum_distance = minimum_distance
        self._friction = friction
        self._controls = controls

        # TODO: remove this hideous thing
        self.smeared = smeared


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


    def _set_controls(self, controls):
        self._controls = controls

    def _get_controls(self):
        if self._controls is not None:
            return self._controls
        else:
            raise ValueError("The controls have not been set.")

    controls = property(_get_controls, _set_controls, "The turbine controls.")

    def force(self, u, tf=None):
        """Return the thrust force exerted by the turbines for given velocity or speed u
        
        :param u: velocity vector or speed
        :type u: dolfin.Function or float
        :param tf: turbine friction function representing one or more turbine. This can also be the integral
              of one or more turbine friction functions, to compute the total force.
        :type tf: dolfin.Function"""
        if tf is None:
            raise TypeError("Turbine friction tf needs to be supplied to compute force")
        return tf * dolfin.dot(u, u)**0.5 * u


    def power(self, u, tf=None):
        """Return the amount of power produced by the turbines for given speed u

        :param u: speed (scalar)
        :type u: dolfin.Function or float
        :param tf: turbine friction function representing one or more turbine. This can also be the integral
              of one or more turbine friction functions, to compute the total power.
        :type tf: dolfin.Function"""
        if tf is None:
            raise TypeError("Turbine friction tf needs to be supplied to compute force")
        return tf * u**3
