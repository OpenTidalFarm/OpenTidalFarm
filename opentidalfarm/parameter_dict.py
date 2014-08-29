class ParameterDictionary(dict):
    '''Parameter dictionary. This subclasses dict so defaults can be set.'''

    def __init__(self, dic={}):
        # Apply dic after defaults so as to overwrite the defaults
        for key, val in dic.iteritems():
            self[key] = val

        self.required = {
            'cost_coef': 'multiplicator that determines the cost per turbine friction',
            'turbine_parametrisation': 'parametrisation of the turbines. If its value is "individual" then the turbines are resolved individually, if "smooth" then the turbines are represented as an average friction over the site area',
            'turbine_pos': 'list of turbine positions',
            'turbine_x': 'turbine extension in the x direction',
            'turbine_y': 'turbine extension in the y direction',
            'turbine_friction': 'turbine friction',
            'rho': 'the density of the fluid',
            'controls': 'a list of the control variables. Valid list values: "turbine_pos" for the turbine position, "turbine_friction" for the friction of the turbine',
            'automatic_scaling': 'activates the initial automatic scaling of the functional',
            'automatic_scaling_multiplier': 'defines the multiplier that determines the initial gradient length (= multiplier * turbine size)',
            'print_individual_turbine_power': 'print out the power output of each individual turbine',
            'output_turbine_power': 'output the power generation of the individual turbines',
            'save_checkpoints': 'automatically store checkpoints after each optimisation iteration',
            'base_path': 'root directory for output',
            'revolve_parameters': '(strategy, snaps_on_disk, snaps_in_ram, verbose)',
             }

    def check(self):
        # First check that no parameters are missing
        for key, error in self.required.iteritems():
            if key not in self:
                raise KeyError('Missing parameter: ' + key + '. ' + 'This is used to set the ' + error + '.')
        # Then check that no parameter is too much (as this is likely to be a mistake!)
        diff = set(self.keys()) - set(self.required.keys())
        if len(diff) > 0:
            raise KeyError('Configuration has too many parameters: ' + str(diff))
