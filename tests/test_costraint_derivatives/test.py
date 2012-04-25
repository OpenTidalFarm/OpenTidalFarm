#!/usr/bin/python
''' This test checks the derivatives of the inequality contraints for the minimal turbine distance constraint. '''
import IPOptUtils
import configuration
import numpy
import sys
from helpers import test_gradient_array
from dolfin import *

config = configuration.DefaultConfiguration()
config.params["controls"] = ['turbine_pos']
f_ieqcons, fprime_ieqcons = IPOptUtils.get_minimum_distance_constraint_func(config)

ieqcons_J = lambda m, forward_only=False: f_ieqcons(m)[0]
ieqcons_dJ = lambda m: fprime_ieqcons(m)[0]
minconv = test_gradient_array(ieqcons_J, ieqcons_dJ, numpy.array([1., 2., 3., 4., 7., 1., 6., 9.]))
if minconv < 1.99:
    info_red("Convergence test failed")
    sys.exit(1)
info_green("Test passed")    
