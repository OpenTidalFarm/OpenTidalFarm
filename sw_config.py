import sw_lib
from turbines import *

from dolfin import * 
from math import exp, sqrt, pi

class Parameters(dict):
    '''Parameter dictionary. This subclasses dict so defaults can be set.'''
    def __init__(self, dict={}):
        # Apply dict after defaults so as to overwrite the defaults
        for key,val in dict.iteritems():
            self[key]=val

        self.required={
            "verbose" : "output verbosity",
            "depth" : "water depth",
            "dt" : "timestep",
            "theta" : "the implicitness for the time discretisation",
            "start_time" : "start time",
            "current_time" : "current time",
            "finish_time" : "finish time",
            "dump_period" : "dump period in timesteps",
            "element_type" : "build the function space",
            'bctype'  : "type of boundary condition to be applied",
            'include_advection': "advection term on",
            'include_diffusion': "diffusion term on",
            'diffusion_coef': "diffusion coefficient",
            'depth' : "water depth in rest",
            'g' : "graviation",
            'k' : "",
            'dump_period' : "dump period",
            'eta0' : "deviantion of the water depth in rest",
            'basin_x' : "length of the basin",
            'basin_y' : "width of the basin",
            'quadratic_friction' : "quadratic friction",
            'friction' : "friction term on",
            'turbine_pos' : "list of turbine positions",
            'turbine_x' : "turbine extension in the x direction",
            'turbine_y' : "turbine extension in the y direction",
            'turbine_friction' : "turbine friction", 
            'controls' : 'the control variables',
            'turbine_model': "turbine model",
            'newton_solver': "newton solver instead of a picard iteration",
            'picard_relative_tolerance': "relative tolerance for the picard iteration",
            'picard_iterations': "maximum number of picard iterations",
            'run_benchmark': "benchmark to compare different solver/preconditioner combinations", 
            'solver_exclude': "solvers/preconditioners to be excluded from the benchmark"
            }

    def check(self):
        # First check that no parameters are missing
        for key, error in self.required.iteritems():
            if not self.has_key(key):
                raise KeyError, "Missing parameter: " + key + ". " + "This is used to set the " + error + "."
        # Then check that no parameter is too much (as this is likely to be a mistake!)
        diff = set(self.keys()) - set(self.required.keys())
        if len(diff) > 0:
            raise KeyError, "Configuration has too many parameters: " + str(diff)

class DefaultConfiguration:
  def __init__(self, nx=20, ny=3, mesh_file=None):
    params = Parameters({
        'verbose'  : 1,
        'element_type'  : sw_lib.p1dgp2,
        'theta' : 0.6,
        'bctype'  : 'flather',
        'include_advection': False,
        'include_diffusion': False,
        'diffusion_coef': 0.0,
        'depth' : 50.,
        'g' : 9.81,
        'dump_period' : 1,
        'eta0' : 2, 
        'basin_x' : 3000.,
        'basin_y' : 1000.,
        'quadratic_friction' : False, 
        'friction' : 0.0, 
        'turbine_pos' : [],
        'turbine_x' : 20., 
        'turbine_y' : 5., 
        'turbine_friction' : [],
        'controls' : ['turbine_pos', 'turbine_friction'],
        'turbine_model': 'BumpTurbine',
        'newton_solver': False, 
        'picard_relative_tolerance': 1e-5, 
        'picard_iterations': 3, 
        'run_benchmark': False, 
        'solver_exclude': ['cg'],
        "start_time": 0.,
        "current_time": 0.,
        "finish_time": 100.
        })

    params["dt"] = params["finish_time"]/4000.
    params["k"] = pi/params['basin_x']

    # Print log messages only from the root process in parallel
    # (See http://fenicsproject.org/documentation/dolfin/dev/python/demo/pde/navier-stokes/python/documentation.html)
    parameters["std_out_all_processes"] = False;

    def generate_mesh(nx, ny):
      ''' Generates a rectangular mesh for the divett test
          nx = Number of cells in x direction
          ny = Number of cells in y direction  '''
      mesh = Rectangle(0, 0, params["basin_x"], params["basin_y"], nx, ny)
      mesh.order()
      mesh.init()
      return mesh

    class Left(SubDomain):
          def inside(self, x, on_boundary):
               return on_boundary and near(x[0], 0.0)

    class Right(SubDomain):
          def inside(self, x, on_boundary):
               return on_boundary and near(x[0], params["basin_x"])

    class Sides(SubDomain):
          def inside(self, x, on_boundary):
               return on_boundary and (near(x[1], 0.0) or near(x[1], params["basin_y"]))

    if mesh_file == None:
        mesh = generate_mesh(nx, ny)
    else:
        mesh = dolfin.Mesh(mesh_file)
    # Initialize sub-domain instances
    left = Left()
    right = Right()
    sides = Sides()

    # Initialize mesh function for boundary domains
    boundaries = FacetFunction("uint", mesh)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    right.mark(boundaries, 2)
    sides.mark(boundaries, 3)
    ds = Measure("ds")[boundaries]

    # Store the result as class variables
    self.params = params
    self.mesh = mesh
    self.ds = ds

  def get_sin_initial_condition(self):

    params = self.params

    class SinusoidalInitialConditions(Expression):
        '''This class implements the Expression class for the shallow water initial condition.'''
        def __init__(self):
            pass
        def eval(self, values, X):
            eta0 = params['eta0']
            g = params['g']
            k = params['k']
            depth = params['depth']
            start_time = params["start_time"]

            values[0] = eta0 * sqrt(g * depth) * cos(k * X[0] - sqrt(g * depth) * k * start_time)
            values[1] = 0.
            values[2] = eta0 * cos(k * X[0] - sqrt(g * depth) * k * start_time)
        def value_shape(self):
            return (3,)
    return SinusoidalInitialConditions
