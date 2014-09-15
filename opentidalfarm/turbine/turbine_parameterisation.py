class TurbineParameterisation(object):
    """Holds turbine parameterisation type information."""
    def __init__(self, default=False, thrust=False, implicit_thrust=False,
                 smeared=False):
        """Sets the parameterisation of the turbines.

        :param default: True if default parameterisation is being used.
        :type default: bool
        :param thrust: True if thrust parameterisation is being used.
        :type thrust: bool
        :param implicit_thrust: True if implicit thrust parameterisation is
            being used.
        :type implicit_thrust: bool
        :param smeared: True if smeared parameterisation is being used.
        :type smeared: bool

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
        """A string naming the parameterisation used."""
        for key in self._parameterisation:
            if self._parameterisation[key]:
                return key


    @property
    def default(self):
        """True if default turbine parameterisation is used."""
        return self._parameterisation["default"]


    @property
    def thrust(self):
        """True if thrust turbine parameterisation is used."""
        return self._parameterisation["thrust"]


    @property
    def implicit_thrust(self):
        """True if implici thrust turbine parameterisation is used."""
        return self._parameterisation["implicit thrust"]


    @property
    def smeared(self):
        """True if smeared turbine parameterisation is used."""
        return self._parameterisation["smeared"]
