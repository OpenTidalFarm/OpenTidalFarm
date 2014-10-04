class ParameterDictionary(dict):
    '''Parameter dictionary. This subclasses dict so defaults can be set.'''

    def __init__(self, dic={}):
        # Apply dic after defaults so as to overwrite the defaults
        for key, val in dic.iteritems():
            self[key] = val

        self.required = {
            'cost_coef': 'multiplicator that determines the cost per turbine friction',
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
