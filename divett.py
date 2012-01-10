from dolfin import * 
from math import exp, sqrt, pi

import sw

params=sw.parameters({
    'depth' : 50.,
    'g' : 9.81,
    'f' : 0.0,
    'dump_period' : 1
    })

# Basin radius.
basin_x=3000 # The length of the basin
basin_y=1000 # The width of the basin
nx=30 # Number of cells in x direction
ny=10 # Number of cells in y direction
# Long wave celerity.
c=sqrt(params["g"]*params["depth"])


params["finish_time"]=100
params["dt"]=params["finish_time"]/4000.

class InitialConditions(Expression):
    def __init__(self):
        pass
    def eval(self, values, X):
        values[0]=0
        values[1]=0.
        values[2]=0.
    def value_shape(self):
        return (3,)


mesh = Rectangle(0, 0, basin_x, basin_y, nx, ny)
mesh.order()
mesh.init()

class Left(SubDomain):
      def inside(self, x, on_boundary):
           return near(x[0], 0.0)

class Right(SubDomain):
      def inside(self, x, on_boundary):
           return near(x[0], basin_x)

# Initialize sub-domain instances
left = Left()
right = Right()

# Initialize mesh function for boundary domains
boundaries = FacetFunction("uint", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
ds = Measure("ds")[boundaries]
