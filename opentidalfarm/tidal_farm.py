import numpy
from dolfin import *
from parameter_dict import ParameterDictionary
from turbines import TurbineCache


class TidalFarm(object):
    ''' A default configuration setup that is used by all tests. '''
    def __init__(self, domain):

        params = ParameterDictionary({
            'turbine_parametrisation': 'individual',
            'turbine_pos': [],
            'turbine_x': 20.,
            'turbine_y': 5.,
            'turbine_friction': [],
            'cost_coef': 0.,
            'controls': ['turbine_pos', 'turbine_friction'],
            'print_individual_turbine_power': False,
            })

        # Store the result as class variables
        self.params = params

        # Create a chaching object for the interpolated turbine friction fields
        # (as their computation is very expensive)
        self.turbine_cache = TurbineCache()

        self.domain = domain

        # Define the subdomain for the turbine site. The default value should
        # only be changed for smeared turbine representations.
        domains = CellFunction("size_t", domain.mesh)
        domains.set_all(1)
        self.site_dx = Measure("dx")[domains]  # The measure used to integrate
                                               # the turbine friction

        # Turbine function space
        self.turbine_function_space = FunctionSpace(self.domain.mesh, 'CG', 2)

    def set_turbine_pos(self, positions, friction=21.0):
        ''' Sets the turbine position and a equal friction parameter. '''
        self.params['turbine_pos'] = positions
        self.params['turbine_friction'] = friction * numpy.ones(len(positions))

    def set_site_dimensions(self, x0, x1, y0, y1):
        ''' Sets the site dimension of a rectangular site. FIXME: move to
        RectangularFarm?'''
        if not x0 < x1 or not y0 < y1:
            raise ValueError("Site must have a positive area")
        self.domain.site_x_start = x0
        self.domain.site_x_end = x1
        self.domain.site_y_start = y0
        self.domain.site_y_end = y1

    def control_array(self):
        ''' This function returns the control parameters as array to be used in
        combination with :class:`ReducedFunctional`. '''
        res = []
        if self.params["turbine_parametrisation"] == "smeared":
            res = numpy.zeros(self.turbine_function_space.dim())

        else:
            if ('turbine_friction' in self.params["controls"] or
                'dynamic_turbine_friction' in self.params["controls"]):
                res += numpy.reshape(self.params['turbine_friction'], -1).tolist()

            if 'turbine_pos' in self.params["controls"]:
                res += numpy.reshape(self.params['turbine_pos'], -1).tolist()

        return numpy.array(res)

