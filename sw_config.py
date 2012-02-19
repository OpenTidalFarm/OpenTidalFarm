import sw_lib
from turbines import *

from dolfin import * 
from math import exp, sqrt, pi


class DefaultConfiguration:
  def __init__(self, nx=20, ny=3):
    params=sw_lib.parameters({
        'basename'  : 'p1dgp2',
        'bctype'  : 'flather',
        'depth' : 50.,
        'g' : 9.81,
        'f' : 0.0,
        'dump_period' : 1,
        'eta0' : 2, # Wave height
        'basin_x' : 3000, # The length of the basin
        'basin_y' : 1000, # The width of the basin
        'friction' : 0.0, # Bottom friction
        'turbine_pos' : [],
        'turbine_length' : 20,
        'turbine_width' : 5,
        'turbine_friction' : 12.0,
        'turbine_model': 'BumpTurbine'
        })

    # Basin radius.
    # Long wave celerity.
    c=sqrt(params["g"]*params["depth"])

    params["finish_time"]=100
    params["dt"]=params["finish_time"]/4000.
    params["k"]=pi/params['basin_x']

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

  def initialise_turbines_measure(self):
    params = self.params

    class Turbines(SubDomain):
          def inside(self, x, on_boundary):
            import numpy
            if len(params["turbine_pos"]) > 0:
              # Check if x lies in a position where a turbine is deployed and if, then increase the friction
              x_pos = numpy.array(params["turbine_pos"])[:,0] 
              x_pos_low = x_pos-params["turbine_length"]/2
              x_pos_high = x_pos+params["turbine_length"]/2

              y_pos = numpy.array(params["turbine_pos"])[:,1] 
              y_pos_low = y_pos-params["turbine_width"]/2
              y_pos_high = y_pos+params["turbine_width"]/2
              if ((x_pos_low <= x[0]) & (x_pos_high >= x[0]) & (y_pos_low <= x[1]) & (y_pos_high >= x[1])).any():
                return True
            return False  

    turbines = Turbines()
    # Initialize mesh function for interior domains
    domains = CellFunction("uint", self.mesh)
    domains.set_all(0)
    turbines.mark(domains, 1)
    self.dx = Measure("dx")[domains]

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
