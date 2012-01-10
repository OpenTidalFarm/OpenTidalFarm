from dolfin import Mesh, Expression
from dolfin.cpp import Rectangle
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

