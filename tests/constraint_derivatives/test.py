#!/usr/bin/python
''' This test checks the derivatives of the inequality contraints for the minimal turbine distance constraint. '''
from opentidalfarm import *
from dolfin import log, INFO
import opentidalfarm.domains
import numpy
import sys
import math

config = configuration.DefaultConfiguration()
domain = opentidalfarm.domains.RectangularDomain(3000, 1000, 20, 3)

config.set_domain(domain)
config.params["controls"] = ['turbine_pos']

ieq = get_minimum_distance_constraint_func(config)

# Only test the correctness of the first inequality constraint for simplicity
ieqcons_J = lambda m: ieq.function(m)[0]
ieqcons_dJ = lambda m, forget=False: ieq.jacobian(m)[0]

minconv = helpers.test_gradient_array(ieqcons_J, ieqcons_dJ, numpy.array([1., 2., 3., 4., 7., 1., 6., 9.]))

if minconv < 1.99:
    info_red("Convergence test for the minimum distance constraints failed")
    sys.exit(1)
info_green("Test passed")    

ieq = generate_site_constraints(config, [[0, 0], [10, 0], [10, 10]], penalty_factor=1)

# Only test the correctness of the first inequality constraint for simplicity
ieqcons_J = lambda m: ieq.function(m)[0]
ieqcons_dJ = lambda m, forget=False: ieq.jacobian(m)[0]

# This will raise a RunetimeWarning, which is fine because we expect
# division by zero
minconv = helpers.test_gradient_array(ieqcons_J, 
             ieqcons_dJ, 
             x=numpy.array([0, 0, 10, 4, 20, 8]), 
             seed=0.1)

# These constraints are linear so we expect no convergence at all.
# Let's check that the tolerance is not above a threshold
log(INFO, "Expecting a Nan convergence order")
if not math.isnan(minconv):
    info_red("Convergence for the polygon shaped sites failed")
    sys.exit(1)
log(INFO, "Test passed")
