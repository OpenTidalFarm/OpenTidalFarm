#!/usr/bin/python
''' This test checks the derivatives of the inequality contraints for the minimal turbine distance constraint. '''
from opentidalfarm import *
from dolfin import log, INFO
import numpy
import math

class TestMinimalDistanceConstraint(object):

    def get_farm(self):
        turbine = BumpTurbine(diameter=10.0, friction=1.0)
        domain = RectangularDomain(0, 0, 3000, 1000, 20, 3)
        farm = Farm(domain, turbine)
        farm.add_turbine((1500, 500))
        return farm

    def test_derivative(self):
        farm = self.get_farm()

        ieq = farm.minimum_distance_constraints()

        # Only test the correctness of the first inequality constraint for simplicity
        ieqcons_J = lambda m: ieq.function(m)[0]
        ieqcons_dJ = lambda m, forget=False: ieq.jacobian(m)[0]

        x = numpy.array([1., 2., 3., 4., 7., 1., 6., 9.])
        minconv = helpers.test_gradient_array(ieqcons_J, ieqcons_dJ, x)

        assert minconv > 1.99

    def test_site_constraint(self):

        farm = self.get_farm()

        ieq = ConvexPolygonSiteConstraint(farm, [[0, 0], [10, 0], [10, 10]])

        # Only test the correctness of the first inequality constraint for
        # simplicity.
        ieqcons_J = lambda m: ieq.function(m)[0]
        ieqcons_dJ = lambda m, forget=False: ieq.jacobian(m)[0]

        # This will raise a RunetimeWarning, which is fine because we expect
        # division by zero
        x = numpy.array([0, 0, 10, 4, 20, 8])
        minconv = helpers.test_gradient_array(ieqcons_J, ieqcons_dJ, x=x,
                                              seed=0.1)

        # These constraints are linear so we expect no convergence at all.
        # Let's check that the tolerance is not above a threshold
        log(INFO, "Expecting a Nan convergence order")
        assert math.isnan(minconv)
