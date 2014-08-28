import numpy
from parameter_dict import ParameterDictionary
from turbines import TurbineCache
from dolfin import *
from math import sqrt, pi
from initial_conditions import *
from domains import *
from helpers import info_red, get_rank
from functionals import DefaultFunctional
import os


class DefaultConfiguration(object):
    ''' A default configuration setup that is used by all tests. '''
    def __init__(self, domain):

        self.functional = DefaultFunctional

        params = ParameterDictionary({
            'turbine_parametrisation': 'individual',
            'turbine_pos': [],
            'turbine_x': 20.,
            'turbine_y': 5.,
            'turbine_friction': [],
            'cost_coef': 0.,
            'rho': 1000.,  # Use the density of water: 1000kg/m^3
            'controls': ['turbine_pos', 'turbine_friction'],
            'automatic_scaling': False,
            'print_individual_turbine_power': False,
            'automatic_scaling_multiplier': 5,
            'output_turbine_power': True,
            'save_checkpoints': False,
            'cache_forward_state': False,
            'base_path': os.curdir,
            'revolve_parameters': None,  # (strategy,
                                         # snaps_on_disk,
                                         # snaps_in_ram,
                                         # verbose)
            })

        # Print log messages only from the root process in parallel
        # (See http://fenicsproject.org/documentation/dolfin/dev/python/demo/pde/navier-stokes/python/documentation.html)
        parameters['std_out_all_processes'] = False

        # Store the result as class variables
        self.params = params

        # Create a chaching object for the interpolated turbine friction fields
        # (as their computation is very expensive)
        self.turbine_cache = TurbineCache()

        # A counter for the current optimisation iteration
        self.optimisation_iteration = 0

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

    def info(self):
        rank = get_rank()

        if rank == 0:
            # Physical parameters
            print "\n=== Physical parameters ==="
            print "Water density: %f kg/m^3" % self.params["rho"]

            # Turbine settings
            print "\n=== Turbine settings ==="
            print "Number of turbines: %i" % len(self.params["turbine_pos"])
            print "Turbines parametrisation: %s" % \
                  self.params["turbine_parametrisation"]
            if self.params["turbine_parametrisation"] == "individual":
                print "Turbines dimensions: %f x %f" % \
                    (self.params["turbine_x"], self.params["turbine_y"])
            print "Control parameters: %s" % ', '.join(self.params["controls"])
            if len(self.params["turbine_friction"]) > 0:
                print "Turbines frictions: %f - %f" % (
                    min(self.params["turbine_friction"]),
                    max(self.params["turbine_friction"]))

            # Optimisation settings
            print "\n=== Optimisation settings ==="
            print "Automatic functional rescaling: %s" % \
                self.params["automatic_scaling"]
            if self.params["automatic_scaling"]:
                print "Automatic functional rescaling multiplier: %s" % \
                    self.params["automatic_scaling_multiplier"]
            print "Automatic checkpoint generation: %s" % \
                self.params["save_checkpoints"]
            print ""

            # Other
            print "\n=== Other ==="
            print "Dolfin version: %s" % dolfin.__version__
            print "Cache forward solution for initial solver guess: %s" % \
                self.params["cache_forward_state"]
            print ""

    def set_site_dimensions(self, site_x_start, site_x_end, site_y_start,
                            site_y_end):
        if not site_x_start < site_x_end or not site_y_start < site_y_end:
            raise ValueError("Site must have a positive area")
        self.domain.site_x_start = site_x_start
        self.domain.site_y_start = site_y_start
        self.domain.site_x_end = site_x_end
        self.domain.site_y_end = site_y_end


class SteadyConfiguration(DefaultConfiguration):
    def __init__(self, domain): 

        super(SteadyConfiguration, self).__init__(domain)

        # Optimisation settings
        self.params['automatic_scaling'] = True

        # Turbine settings
        self.params['turbine_pos'] = []
        self.params['turbine_friction'] = []
        self.params['turbine_x'] = 20.
        self.params['turbine_y'] = 20.
        self.params['controls'] = ['turbine_pos']

        # Finally set some DOLFIN optimisation flags
        dolfin.parameters['form_compiler']['cpp_optimize'] = True
        dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'
        dolfin.parameters['form_compiler']['optimize'] = True


class UnsteadyConfiguration(SteadyConfiguration):
    def __init__(self, domain): 
        super(UnsteadyConfiguration, self).__init__(domain)
