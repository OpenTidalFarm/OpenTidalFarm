import sw_lib
from turbines import *

from dolfin import * 
from math import exp, sqrt, pi


class DefaultConfiguration:
  def __init__(self, nx=20, ny=3):
    params=sw_lib.parameters({
        'basename'  : 'p1dgp2',
        'bctype'  : 'flather',
        'include_advection': False,
        'include_diffusion': False,
        'diffusion_coef': 0.0,
        'depth' : 50.,
        'g' : 9.81,
        'f' : 0.0,
        'dump_period' : 1,
        'eta0' : 2, # Wave height
        'basin_x' : 3000., # The length of the basin
        'basin_y' : 1000., # The width of the basin
        'quadratic_friction' : False, 
        'friction' : 0.0, # Bottom friction
        'turbine_pos' : [],
        'turbine_x' : 20., # The turbine extension in the x direction
        'turbine_y' : 5., # The turbine extension in the y direction
        'turbine_friction' : 12.0,
        'turbine_model': 'BumpTurbine',
        'newton_solver': False, # Only used with quadratic friction: Use a Newton solver to solve the nonlinear problem. The default is to use Picard iterations.
        'picard_iterations': 3, # If quadratic_friction is True and newton_solver is False, then this many picard iterations are performed to solve the nonlinear problem.
        'run_benchmark': False, # If true, then a combination of solver/preconditioner variations are used for each solve and their timings reported
        'solver_exclude': [] # A list of solvers that are to be excluded from the benchmark test 
        })

    # Basin radius.
    # Long wave celerity.
    c=sqrt(params["g"]*params["depth"])

    params["start_time"] = 0
    params["finish_time"] = 100
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

    mesh = generate_mesh(nx, ny)
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
