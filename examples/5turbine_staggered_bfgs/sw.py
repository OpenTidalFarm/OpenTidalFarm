''' This example optimises the position of three turbines using the hallow water model. '''

import sys
import configuration 
import numpy
import IPOptUtils
from dirichlet_bc import DirichletBCSet
from helpers import test_gradient_array
from reduced_functional import ReducedFunctional
from dolfin import *
from scipy.optimize import fmin_l_bfgs_b
from helpers import info, info_green, info_red, info_blue
set_log_level(ERROR)


# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.ConstantInflowPeriodicSidesPaperConfiguration(nx=100, ny=33)
config.params["automatic_scaling"] = True

# The turbine position is the control variable 
turbine_pos = [[60, 38], [80, 28], [100, 38], [120, 28], [140, 38]] 
config.set_turbine_pos(turbine_pos)
info_blue("Deployed " + str(len(turbine_pos)) + " turbines.")

model = ReducedFunctional(config, scaling_factor = -1, plot = True)
m0 = model.initial_control()

# Get the upper and lower bounds for the turbine positions
lb, ub = IPOptUtils.position_constraints(config)
bounds = [(lb[i], ub[i]) for i in range(len(lb))]
fmin_l_bfgs_b(model.j, m0, fprime = model.dj, bounds = bounds, iprint = 2)
