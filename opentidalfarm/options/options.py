class OptionsManager(dict):
    """A global output management dictionary."""

    def __init__(self, dictionary={}):
        for key, value in self._defaults.iteritems():
            self[key] = value

        # Apply dictionary after defaults so as to overwrite the defaults.
        for key, value in dictionary.iteritems():
            self[key] = value

    @property
    def descriptions(self):
        return {
            "save_checkpoints": "Save checkpoints.",
            "output_total_power": "The total power extracted from the farm(s).",
            "output_individual_power": ("The power extracted by individual "
                                        "turbines."),
            "dump_period": ("The number of iterations between dumping the "
                            "solution to disk.")
        }


    @property
    def _defaults(self):
        return {
            "save_checkpoints": False,
            "output_total_power": True,
            "output_individual_power": False,
            "dump_period": 1
        }


    def __str__(self):
        string = "Option:".rjust(20) + " [Value] Description [Default]"
        defaults = self._defaults
        descriptions = self.descriptions
        for key, val in self.iteritems():
            string += ("\n" + (key + ":").rjust(20) +
                       " [" + str(val) + "] " +
                       descriptions[key] +
                       "[Default: " + str(defaults[key]) + "]")
        return string


# Initialize an instance of the OptionsManager class.
options = OptionsManager()
