from dolfin import * 
from math import exp, sqrt, pi

import sw_lib

class SWConfiguration:
  def __init__(self, nx=20, ny=3):
    params=sw_lib.parameters({
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
        'turbine_friction' : 12.0
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
