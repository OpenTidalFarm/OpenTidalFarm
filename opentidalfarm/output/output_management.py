class OutputManagement(dict):
    """A global output management dictionary."""

    def __init__(self, dictionary={}):
        for key, value in self._defaults.iteritems():
            self[key] = value

        # Apply dictionary after defaults so as to overwrite the defaults.
        for key, value in dictionary.iteritems():
            self[key] = value

    @property
    def options(self):
        return {
            "total_power": ("The total power extracted from the farm(s). "
                            "[Default: True]"),
            "individual_power": ("The power extracted by individual turbines. "
                                 "[Default: False]"),
        }


    @property
    def _defaults(self):
        return {
            "total_power": True,
            "individual_power": False
        }


    def __str__(self):
        string = "Option:".rjust(20) + " [Value] Description"
        for key, val in self.iteritems():
            string += ("\n" + (key + ":").rjust(20) + " [" + str(val) + "] " +
                       self.options[key])
        return string


# Initialize an instance of the OutputManagement class.
output_options = OutputManagement()
