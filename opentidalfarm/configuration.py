import finite_elements
import numpy
from parameter_dict import ParameterDictionary
from dirichlet_bc import DirichletBCSet
from turbines import TurbineCache
from dolfin import *
from math import exp, sqrt, pi
from initial_conditions import *
from domains import *
from helpers import info, info_green, info_red, info_blue
from functionals import DefaultFunctional
import os

class DefaultConfiguration(object):
  ''' A default configuration setup that is used by all tests. '''
  def __init__(self, nx=20, ny=3, basin_x = 3000, basin_y = 1000, finite_element = finite_elements.p2p1):

    # Initialize function space and the domain
    self.finite_element = finite_element
    self.set_domain(RectangularDomain(basin_x, basin_y, nx, ny), warning = False)

    self.functional = DefaultFunctional

    params = ParameterDictionary({
        'verbose'  : 1,
        'theta' : 0.6,
        'steady_state' : False,
        'functional_final_time_only' : False,
        'functional_quadrature_degree' : 1,
        'bctype'  : 'flather',
        'strong_bc' : None,
        'free_slip_on_sides' : False,
        'include_advection': False,
        'include_diffusion': False,
        'include_time_term': True,
        'diffusion_coef': 0.0,
        'depth' : 50.0,
        'g' : 9.81,
        'dump_period' : 1,
        'eta0' : 2,
        'quadratic_friction' : False,
        'friction' : Constant(0.0),
        'turbine_parametrisation' : 'individual',
        'turbine_pos' : [],
        'turbine_x' : 20.,
        'turbine_y' : 5.,
        'turbine_friction' : [],
        'cost_coef': 0.,
        'turbine_thrust_parametrisation' : False,
        'implicit_turbine_thrust_parametrisation' : False,
        'rho' : 1000., # Use the density of water: 1000kg/m^3
        'controls' : ['turbine_pos', 'turbine_friction'],
        'newton_solver': False,
        'linear_solver' : 'mumps',
        'preconditioner' : 'default',
        'picard_relative_tolerance': 1e-5,
        'picard_iterations': 3,
        'run_benchmark': False,
        'solver_exclude': ['cg'],
        'start_time': 0.,
        'current_time': 0.,
        'finish_time': 100.,
        'automatic_scaling': False,
        'print_individual_turbine_power': False,
        'automatic_scaling_multiplier': 5,
        'output_turbine_power': True,
        'save_checkpoints': False,
        'base_path': os.curdir
        })

    params['dt'] = params['finish_time']/4000.

    # Print log messages only from the root process in parallel
    # (See http://fenicsproject.org/documentation/dolfin/dev/python/demo/pde/navier-stokes/python/documentation.html)
    parameters['std_out_all_processes'] = False

    params['k'] = pi/self.domain.basin_x

    # Store the result as class variables
    self.params = params

    params['initial_condition'] = ConstantFlowInitialCondition(self)

    # Create a chaching object for the interpolated turbine friction fields (as their computation is very expensive)
    self.turbine_cache = TurbineCache()

    # A counter for the current optimisation iteration
    self.optimisation_iteration = 0

  def set_domain(self, domain, warning = True):
      if warning:
           info_red("If you are overwriting the domain, make sure that you reapply the boundary conditions as well")
      self.domain = domain

      # Define the subdomain for the turbine site. The default value should only be changed for smooth turbine representations.
      domains = CellFunction("size_t", self.domain.mesh)
      domains.set_all(1)
      self.site_dx = Measure("dx")[domains] # The measure used to integrate the turbine friction

      V, H = self.finite_element(self.domain.mesh)
      T = FunctionSpace(self.domain.mesh, 'CG', 2)              # Turbine space

      self.turbine_function_space = T
      self.function_space = MixedFunctionSpace([V, H])
      self.function_space_enriched = MixedFunctionSpace([V, H, T])
      self.function_space_2enriched = MixedFunctionSpace([V, H, T, T])

  def set_turbine_pos(self, positions, friction = 21.0):
      ''' Sets the turbine position and a equal friction parameter. '''
      self.params['turbine_pos'] = positions
      self.params['turbine_friction'] = friction * numpy.ones(len(positions))

  def info(self):
    hmin = MPI.min(self.domain.mesh.hmin())
    hmax = MPI.max(self.domain.mesh.hmax())
    if MPI.process_number() == 0:
        print "\n=== Physical parameters ==="
        if isinstance(self.params["depth"],float):
          print "Water depth: %f m" % self.params["depth"]
        print "Gravity constant: %f m/s^2" % self.params["g"]
        print "Viscosity constant: %f m^2/s" % self.params["diffusion_coef"]
        print "Water density: %f kg/m^3" % self.params["rho"]
        if isinstance(self.params["friction"], dolfin.functions.constant.Constant):
            print "Bottom friction: %s" % (self.params["friction"](0))
        else:
            print "Bottom fruction: %f - %f" %\
                (self.params["friction"].vector().array().min(),\
                 self.params["friction"].vector().array().max())
        print "Advection term: %s" % self.params["include_advection"]
        print "Diffusion term: %s" % self.params["include_diffusion"]
        print "Steady state: %s" % self.params["steady_state"]
        print "\n=== Turbine settings ==="
        print "Number of turbines: %i" % len(self.params["turbine_pos"])
        print "Turbines parametrisation: %s" % self.params["turbine_parametrisation"]
        if self.params["turbine_parametrisation"] == "individual":
            print "Turbines dimensions: %f x %f" % (self.params["turbine_x"], self.params["turbine_y"])
        print "Control parameters: %s" % ', '.join(self.params["controls"])
        if len(self.params["turbine_friction"]) > 0:
          print "Turbines frictions: %f - %f" % (min(self.params["turbine_friction"]), max(self.params["turbine_friction"]))
        print "\n=== Discretisation settings ==="
        print "Finite element pair: ", self.finite_element.func_name
        print "Steady state: ", self.params["steady_state"]
        if not self.params["steady_state"]:
            print "Theta: %f" % self.params["theta"]
            print "Start time: %f s" % self.params["start_time"]
            print "Finish time: %f s" % self.params["finish_time"]
            print "Time step: %f s" % self.params["dt"]
        print "Number of mesh elements: %i" % self.domain.mesh.num_cells()
        print "Mesh element size: %f - %f" % (hmin, hmax)
        print "\n=== Optimisation settings ==="
        print "Automatic functional rescaling: %s" % self.params["automatic_scaling"]
        if self.params["automatic_scaling"]:
          print "Automatic functional rescaling multiplier: %s" % self.params["automatic_scaling_multiplier"]
        print "Automatic checkpoint generation: %s" % self.params["save_checkpoints"]
        print ""

  def set_site_dimensions(self, site_x_start, site_x_end, site_y_start, site_y_end):
      if not site_x_start < site_x_end or not site_y_start < site_y_end:
          raise ValueError, "Site must have a positive area"
      self.domain.site_x_start = site_x_start
      self.domain.site_y_start = site_y_start
      self.domain.site_x_end = site_x_end
      self.domain.site_y_end = site_y_end

class SteadyConfiguration(DefaultConfiguration):
    def __init__(self, mesh_file, inflow_direction, finite_element=finite_elements.p2p1):

        super(SteadyConfiguration, self).__init__(finite_element=finite_element)
        # Model settings
        self.set_domain(GMeshDomain(mesh_file), warning=False)
        self.params['steady_state'] = True
        self.params['initial_condition'] = ConstantFlowInitialCondition(self)
        self.params['include_advection'] = True
        self.params['include_diffusion'] = True
        self.params['diffusion_coef'] = 3.0
        self.params['quadratic_friction'] = True
        self.params['newton_solver'] = True
        self.params['friction'] = Constant(0.0025)
        self.params['theta'] = 1.0

        # Boundary conditions
        bc = DirichletBCSet(self)
        bc.add_constant_flow(1, 2.0+1e-10, direction=inflow_direction)
        bc.add_zero_eta(2)
        self.params['bctype'] = 'strong_dirichlet'
        self.params['strong_bc'] = bc
        self.params['free_slip_on_sides'] = True

        # Optimisation settings
        self.params['functional_final_time_only'] = True
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
    def __init__(self, mesh_file, inflow_direction, finite_element = finite_elements.p2p1, period = 12.*60*60, eta0=2.0):
        super(UnsteadyConfiguration, self).__init__(mesh_file, inflow_direction, finite_element)

        # Switch to the unsteady shallow water equations
        self.params['steady_state'] = False
        self.params['functional_final_time_only'] = False

        # Timing settings
        self.params['theta'] = 1.0
        self.params['start_time'] = 1./4*period
        self.params['dt'] = period/50
        self.params['finish_time'] = 5./4*period

        # Initial condition
        k = 2*pi/(period*sqrt(self.params['g']*self.params['depth']))
        info('Wave period (in h): %f' % (period/60/60) )
        info('Approximate CFL number (assuming a velocity of 2): ' + str(2*self.params['dt']/self.domain.mesh.hmin()))
        self.params['initial_condition'] = SinusoidalInitialCondition(self, eta0, k, self.params['depth'])

        # Boundary conditions
        bc = DirichletBCSet(self)
        expression = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), eta0 = eta0, g = self.params["g"], depth = self.params["depth"], t = self.params["current_time"], k = k)
        bc.add_analytic_u(1, expression)
        bc.add_analytic_u(2, expression)
        bc.add_noslip_u(3)
        self.params['strong_bc'] = bc
