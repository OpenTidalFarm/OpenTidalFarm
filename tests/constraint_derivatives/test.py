#!/usr/bin/python
''' This test checks the derivatives of the inequality contraints for the minimal turbine distance constraint. '''
from opentidalfarm import *
import numpy
import sys

config = configuration.DefaultConfiguration()
config.params["controls"] = ['turbine_pos']
ieq = get_minimum_distance_constraint_func(config)

ieqcons_J = lambda m: ieq['fun'](m)[0]
ieqcons_dJ = lambda m, forget=False: ieq['jac'](m)[0]
minconv = helpers.test_gradient_array(ieqcons_J, ieqcons_dJ, numpy.array([1., 2., 3., 4., 7., 1., 6., 9.]))
if minconv < 1.99:
    info_red("Convergence test for the minimum distance constraints failed")
    sys.exit(1)
info_green("Test passed")    

ieq = generate_site_constraints(config, [[10, 10], [20, 10], [20, 20], [10, 20]])
ieqcons_J = lambda m: ieq['fun'](m)[0]
ieqcons_dJ = lambda m, forget=False: ieq['jac'](m)[0]
minconv = helpers.test_gradient_array(ieqcons_J, ieqcons_dJ, numpy.array([1., 2., 3., 4., 7., 1., 6., 9.]))
if minconv < 1.99:
    info_red("Convergence for the polygon shaped sites failed")
    sys.exit(1)
info_green("Test passed")    
