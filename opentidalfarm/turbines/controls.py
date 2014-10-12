class Controls(object):
    """Holds the controls for optimisation.

    This class holds the controls for optimisation, such as the position and the
    friction of the turbines. The user initializes this class with their desired
    control parameters.
    """
    def __init__(self, position=False, friction=False, dynamic_friction=False):
        """Initialize with the desired controls.

        :param bool position: Whether or not turbine position is a control.
        :param bool friction: Whether or not turbine friction is a control.
        :param bool dynamic_friction: Whether or not dynamic friction is a control.
        """

        self._controls = {"position": False,
                          "friction": False,
                          "dynamic_friction": False}

        def _process(key, value):
            """Check value is of type bool. Raise ValueError if it is not."""
            try:
                assert isinstance(value, bool)
                # Change the control value in the dictionary.
                self._controls[key] = value
            # Raise an error if a boolean was not given.
            except AssertionError:
                raise ValueError("%s must be a boolean (%s)." %
                                 (key.capitalize(), str(type(value))))

        # Process the given values
        _process("position", position)
        _process("friction", friction)
        _process("dynamic friction", dynamic_friction)


    def __str__(self):
        """Returns a string representation of the enabled control parameters."""
        string = "Control parameters:"
        # Get enabled controls.
        enabled = [key for key in self._controls if self._controls[key]]
        # Add the keys to the string to be returned.
        if len(enabled) > 0:
            for control in enabled:
                string += "\n - %s" % control.capitalize()
        else:
            string += " no control parameters have been enabled!"
        return string


    @property
    def position(self):
        """Whether position is enabled as a control parameter.

        :returns: True if position is enabled as a control parameter.
        :rtype: bool
        """
        return self._controls["position"]


    @property
    def friction(self):
        """Whether friction is enabled as a control parameter.

        :returns: True if friction is enabled as a control parameter.
        :rtype: bool
        """
        return self._controls["friction"]


    @property
    def dynamic_friction(self):
        """Whether dynamic friction is enabled as a control parameter.

        :returns: True if dynamic friction is enabled as a control parameter.
        :rtype: bool
        """
        return self._controls["dynamic friction"]
