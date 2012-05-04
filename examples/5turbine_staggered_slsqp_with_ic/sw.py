''' This example optimises the position of three turbines using the shallow water model. '''

import sys
import configuration 
import numpy
import IPOptUtils
from helpers import test_gradient_array
from reduced_functional import ReducedFunctional
from dolfin import *
from scipy.optimize import fmin_slsqp
set_log_level(ERROR)

parameters["std_out_all_processes"] = False

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.ConstantInflowPeriodicSidesPaperConfiguration()

# The turbine position is the control variable 
turbine_pos = [[60, 38], [80, 28], [100, 38], [120, 28], [140, 38]] 

config.set_turbine_pos(turbine_pos)
info_blue("Deployed " + str(len(turbine_pos)) + " turbines.")

model = ReducedFunctional(config, scaling_factor = -10**-1, plot = True)
m0 = model.initial_control()

g = lambda m: []
dg = lambda m: []

# Get the upper and lower bounds for the turbine positions
lb, ub = IPOptUtils.position_constraints(config)
bounds = [(lb[i], ub[i]) for i in range(len(lb))]

f_ieqcons, fprime_ieqcons = IPOptUtils.get_minimum_distance_constraint_func(config)

fmin_slsqp(model.j, m0, fprime = model.dj, bounds = bounds, f_ieqcons = f_ieqcons, fprime_ieqcons = fprime_ieqcons, iprint = 2, full_output = True)
