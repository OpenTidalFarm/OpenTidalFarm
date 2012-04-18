import finite_elements
from dirichlet_bc import DirichletBCSet
from dolfin import * 
from math import exp, sqrt, pi
from initial_conditions import *
from domains import *

class Parameters(dict):
    '''Parameter dictionary. This subclasses dict so defaults can be set.'''
    def __init__(self, dict={}):
        # Apply dict after defaults so as to overwrite the defaults
        for key,val in dict.iteritems():
            self[key]=val

        self.required={
            'verbose' : 'output verbosity',
            'dt' : 'timestep',
            'theta' : 'the implicitness for the time discretisation',
            'start_time' : 'start time',
            'current_time' : 'current time',
            'finish_time' : 'finish time',
            'dump_period' : 'dump period in timesteps',
            'bctype'  : 'type of boundary condition to be applied',
            'strong_bc'  : 'list of strong dirichlet boundary conditions to be applied',
            'initial_condition'  : 'initial condition function',
            'include_advection': 'advection term on',
            'include_diffusion': 'diffusion term on',
            'diffusion_coef': 'diffusion coefficient',
            'depth' : 'water depth at rest',
            'g' : 'graviation',
            'k' : 'wave length paramter. If you want a wave lenght of l, then set k to 2*pi/l.',
            'dump_period' : 'dump period',
            'eta0' : 'deviantion of the water depth in rest',
            'quadratic_friction' : 'quadratic friction',
            'friction' : 'friction term on',
            'turbine_pos' : 'list of turbine positions',
            'turbine_x' : 'turbine extension in the x direction',
            'turbine_y' : 'turbine extension in the y direction',
            'turbine_friction' : 'turbine friction', 
            'functional_turbine_scaling' : 'scaling of the turbine dimensions in the functional, which often gives a better posted problem',
            'controls' : 'the control variables',
            'turbine_model': 'turbine model',
            'newton_solver': 'newton solver instead of a picard iteration',
            'linear_solver' : 'default linear solver',
            'preconditioner' : 'default preconditioner',
            'picard_relative_tolerance': 'relative tolerance for the picard iteration',
            'picard_iterations': 'maximum number of picard iterations',
            'run_benchmark': 'benchmark to compare different solver/preconditioner combinations', 
            'solver_exclude': 'solvers/preconditioners to be excluded from the benchmark'
            }

    def check(self):
        # First check that no parameters are missing
        for key, error in self.required.iteritems():
            if not self.has_key(key):
                raise KeyError, 'Missing parameter: ' + key + '. ' + 'This is used to set the ' + error + '.'
        # Then check that no parameter is too much (as this is likely to be a mistake!)
        diff = set(self.keys()) - set(self.required.keys())
        if len(diff) > 0:
            raise KeyError, 'Configuration has too many parameters: ' + str(diff)

class DefaultConfiguration(object):
  def __init__(self, nx=20, ny=3, basin_x = 3000, basin_y = 1000, mesh_file=None, finite_element = finite_elements.p2p1):
    self.domain = RectangularDomain(basin_x, basin_y, nx, ny)
    params = Parameters({
        'verbose'  : 1,
        'theta' : 0.6,
        'initial_condition' : SinusoidalInitialCondition, 
        'bctype'  : 'flather',
        'strong_bc' : None,
        'include_advection': False,
        'include_diffusion': False,
        'diffusion_coef': 0.0,
        'depth' : 50.,
        'g' : 9.81,
        'dump_period' : 1,
        'eta0' : 2, 
        'quadratic_friction' : False, 
        'friction' : 0.0, 
        'turbine_pos' : [],
        'turbine_x' : 20., 
        'turbine_y' : 5., 
        'turbine_friction' : [],
        'functional_turbine_scaling' : 0.5,
        'controls' : ['turbine_pos', 'turbine_friction'],
        'turbine_model': 'BumpTurbine',
        'newton_solver': False, 
        'linear_solver' : 'default',
        'preconditioner' : 'default',
        'picard_relative_tolerance': 1e-5, 
        'picard_iterations': 3, 
        'run_benchmark': False, 
        'solver_exclude': ['cg'],
        'start_time': 0.,
        'current_time': 0.,
        'finish_time': 100.
        })

    params['dt'] = params['finish_time']/4000.

    # Print log messages only from the root process in parallel
    # (See http://fenicsproject.org/documentation/dolfin/dev/python/demo/pde/navier-stokes/python/documentation.html)
    parameters['std_out_all_processes'] = False;

    params['k'] = pi/self.domain.basin_x

    # Initialize function space
    self.finite_element = finite_element
    self.function_space = finite_element(self.domain.mesh) 

    # Store the result as class variables
    self.params = params

class PaperConfiguration(DefaultConfiguration):
  def __init__(self, nx = 20, ny = 3, basin_x = None, basin_y = None, finite_element = finite_elements.p2p1):
    # If no mesh file is given, we compute the domain size from the number of elements(nx and ny) with a 2m element size
    if not basin_x:
      basin_x = float(nx * 2) # Use a 2m element size by default
    if not basin_y:
      basin_y = float(ny * 2)

    super(PaperConfiguration, self).__init__(nx, ny, basin_x, basin_y, mesh_file, finite_element)

    # Model settings
    self.params['include_advection'] = True
    self.params['include_diffusion'] = True
    self.params['diffusion_coef'] = 2.0
    self.params['quadratic_friction'] = True
    self.params['newton_solver'] = True 
    self.params['friction'] = 0.0025
    #self.params['eta0'] = 2 * sqrt(self.params["depth"]/self.params["g"]) # This will give a inflow velocity of 2m/s

    # Turbine settings
    self.params['turbine_pos'] = []
    self.params['turbine_friction'] = []
    self.params['turbine_x'] = 20. 
    self.params['turbine_y'] = 20. 
    self.params['functional_turbine_scaling'] = 0.5
    self.params['controls'] = ['turbine_pos']
    self.params['turbine_model'] = 'BumpTurbine'

    # Timing settings
    self.period = 1.24*60*60 
    self.params['k'] = 2*pi/(self.period*sqrt(self.params['g']*self.params['depth']))
    self.params['theta'] = 1.0
    self.params['start_time'] = 1./4*self.period
    self.params['dt'] = self.period/50
    self.params['finish_time'] = 3./4*self.period
    info('Wave period (in h): %f' % (self.period/60/60) )
    info('Approximate CFL number (assuming a velocity of 2): ' +str(2*self.params['dt']/self.mesh.hmin()))

    # Configure the boundary conditions
    self.params['bctype'] = 'dirichlet',
    self.params['bctype'] = 'strong_dirichlet'
    bc = DirichletBCSet(self)
    bc.add_analytic_u(self.domain.left)
    bc.add_analytic_u(self.domain.right)
    bc.add_noslip_u(self.domain.sides)
    self.params['strong_bc'] = bc

    # Finally set some optimistion flags 
    dolfin.parameters['form_compiler']['cpp_optimize'] = True
    dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

class ConstantInflowPeriodicSidesPaperConfiguration(PaperConfiguration):
  def __init__(self, nx = 20, ny = 3, basin_x = None, basin_y = None, mesh_file = None, finite_element = finite_elements.p2p1):
    super(ConstantInflowPeriodicSidesPaperConfiguration, self).__init__(nx, ny, basin_x, basin_y, mesh_file, finite_element)

    self.params["initial_condition"] = ConstantFlowInitialCondition 
    self.params["newton_solver"] = False
    self.params["picard_iterations"] = 2
    self.params['theta'] = 1.0

    bc = DirichletBCSet(self)
    bc.add_constant_flow(self.domain.left)
    bc.add_noslip_u(self.domain.sides)
    self.params['strong_bc'] = bc

    self.params['start_time'] = 0.0
    self.params['dt'] = self.period
    self.params['finish_time'] = self.params['start_time'] + self.params['dt'] 

