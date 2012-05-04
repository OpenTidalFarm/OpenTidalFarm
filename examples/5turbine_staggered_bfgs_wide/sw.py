''' This example optimises the position of three turbines using the hallow water model. '''

import sys
import configuration 
import numpy
import IPOptUtils
from helpers import test_gradient_array
from reduced_functional import ReducedFunctional
from dolfin import *
from scipy.optimize import fmin_l_bfgs_b
set_log_level(ERROR)

parameters["std_out_all_processes"] = False

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.WideConstantInflowPeriodicSidesPaperConfiguration()

# The turbine position is the control variable
offset = 33.
turbine_pos = [[60, 38 + offset], [80, 28 + offset], [100, 38 + offset], [120, 28 + offset], [140, 38 + offset]] 

config.set_turbine_pos(turbine_pos)
info_blue("Deployed " + str(len(turbine_pos)) + " turbines.")

model = ReducedFunctional(config, scaling_factor = -10**1, plot = True)
m0 = model.initial_control()

g = lambda m: []
dg = lambda m: []

# Get the upper and lower bounds for the turbine positions
lb, ub = IPOptUtils.position_constraints(config, spacing_sides = 66./2, spacing_left = 50.)
print "lb", lb
print "ub", ub
bounds = [(lb[i], ub[i]) for i in range(len(lb))]
fmin_l_bfgs_b(model.j, m0, fprime = model.dj, bounds = bounds)
