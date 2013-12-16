#!/usr/bin/python
''' This test checks the derivatives of the inequality contraints for the domain constraint. '''
from opentidalfarm import *
import opentidalfarm.domains
import numpy
import sys
import math

config = configuration.DefaultConfiguration(nx=100, ny=50)
config.set_domain(opentidalfarm.domains.RectangularDomain(3000, 1000, 100, 50))
config.params["controls"] = ['turbine_pos']

class TurbineSite(SubDomain):
  def inside(self, x, on_boundary):
    return True if 1300 <= x[0] <= 1700 and 300 <= x[1] <= 700 else False

domains = MeshFunction('size_t', config.domain.mesh, 2)
domains.set_all(0)
turbine_site = TurbineSite()
turbine_site.mark(domains, 1)

feasible_area = get_distance_function(config, domains)

ieq = get_domain_constraints(config, feasible_area, attraction_center=((1500, 500)))

# Test the case where turbines are outside the domain
ieqcons_J = lambda m: ieq.function(m)[0]
ieqcons_dJ = lambda m, forget=False: ieq.jacobian(m)[0]
minconv = helpers.test_gradient_array(ieqcons_J, ieqcons_dJ, numpy.array([-10., -10., -100, 80]))

if minconv < 1.99:
    info_red("Convergence test for the minimum distance constraints failed")
    sys.exit(1)

# Now test the case where the turbines are inside 
# The normal Taylor test does not work here, so lets just compare the gradient with a finite difference approach
x = numpy.array([100., 900.])
hx = numpy.array([1., 0.])
eps = 2.
grad = ieqcons_dJ(x)
assert (((ieqcons_J(x+eps*hx) - ieqcons_J(x))/eps) - grad[0])/grad[0] < 1e-10

x = numpy.array([1500., 250.])
hy = numpy.array([0., 1.])
eps = 2.
grad = ieqcons_dJ(x)
assert (((ieqcons_J(x+eps*hy) - ieqcons_J(x))/eps) - grad[1])/grad[1] < 1e-10

info_green("Test passed")
