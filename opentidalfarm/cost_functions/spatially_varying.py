# Produce a dolfin expression from discrete data and project it onto a functionspace

import numpy
import scipy.interpolate
from dolfin import *

# create mesh and functionspace
mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, "CG", 1)

# create some random data
x = numpy.linspace(0, 1, 5)
y = numpy.linspace(0, 1, 5)
X, Y = numpy.meshgrid(x,y)

def somefunc(x, y):
  #return x*y
  #return 2.0 - 1.0 * ( x**2 + y**2 ) - 0.5*y
  z = []
  for i in range(len(x)): z.append(numpy.random.uniform(-20., 20.0, len(x)))
  return numpy.array(z)

Z = somefunc(X, Y)

# interpolate it
s = scipy.interpolate.RectBivariateSpline(x,y,Z)

# this is the crucial bit!
class interpolatedExpression(Expression):
    def __init__(self, s):
      self.s = s

    def eval(self, value, x):
      value[0] = self.s(x[0], x[1])

# create an instance of this expression
expr = interpolatedExpression(s)
# and project it onto a function space
f = project(expr, V)

def differentiate(x, y, s):
    x1, y1 = x - (x*0.0001), y - (y*0.0001)
    x2, y2 = x + (x*0.0001), y + (y*0.0001)
    z1, z2 = float(s(x1,y1)), float(s(x2,y2))
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return numpy.array([dz / dx, dz / dy])

# test it
x = 0.6667
y = 0.8412

print differentiate(x, y, s)
# dolfin function
print f(x,y)
# scipy interpolated function
print s(x,y)

plot(f, interactive = True)