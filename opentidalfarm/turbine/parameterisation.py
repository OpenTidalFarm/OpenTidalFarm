class TurbineParameterisation(object):
    """Holds turbine parameterisation type information."""
    def __init__(self, default=False, thrust=False, implicit_thrust=False,
                 smeared=False):
        """Sets the parameterisation of the turbines.

        :param bool default: True if default parameterisation is being used.
        :param bool thrust: True if thrust parameterisation is being used.
        :param bool implicit_thrust: True if implicit thrust parameterisation is being used.
        :param bool smeared: True if smeared parameterisation is being used.
        """
        parameterisation = [default, thrust, implicit_thrust, smeared]

        # Ensure default parameterisation if no other parameterisation is given.
        if parameterisation.count(True)==0:
            default = True

        else:
            try:
                assert(parameterisation.count(True)==1)
            except AssertionError:
                raise ValueError("Turbines may only have a single "
                                 "parameterisation!")

        # Store the parameterisation in a dictionary so we can call
        # TurbineParamaterisation.parameterisation to get a string
        # representation of the type of parameterisation.
        self._parameterisation = {"default": default,
                                  "thrust": thrust,
                                  "implicit thrust": implicit_thrust,
                                  "smeared": smeared}


    def __str__(self):
        """Returns a string containing the parameterisation of the turbine."""
        return self.parameterisation


    @property
    def parameterisation(self):
        """A string naming the parameterisation used.
        :returns: The type of parameterisation being used.
        :rtype: str
        """
        for key in self._parameterisation:
            if self._parameterisation[key]:
                return key


    @property
    def default(self):
        """True if default turbine parameterisation is used.
        :returns: Whether default turbine parameterisation is used.
        :rtype: bool
        """
        return self._parameterisation["default"]


    @property
    def thrust(self):
        """True if thrust turbine parameterisation is used.
        :returns: Whether thrust parameterisation is used.
        :rtype: bool
        """
        return self._parameterisation["thrust"]


    @property
    def implicit_thrust(self):
        """True if implicit thrust turbine parameterisation is used.
        :returns: Whether implicit thrust parameterisation is used.
        :rtype: bool
        """
        return self._parameterisation["implicit thrust"]


    @property
    def smeared(self):
        """True if smeared turbine parameterisation is used.
        :returns: Whether smeared parameterisation is used.
        :rtype: bool
        """
        return self._parameterisation["smeared"]
