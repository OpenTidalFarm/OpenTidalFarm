''' This example optimises the position of three turbines using the hallow water model. '''

import sys
import configuration 
import numpy
from dirichlet_bc import DirichletBCSet
from helpers import test_gradient_array
from animated_plot import *
from reduced_functional import ReducedFunctional
from dolfin import *
from scipy.optimize import fmin_l_bfgs_b
set_log_level(ERROR)

# An animated plot to visualise the development of the functional value
plot = AnimatedPlot(xlabel='Iteration', ylabel='Functional value')

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.ConstantInflowPeriodicSidesPaperConfiguration(nx=100, ny=33)

# The turbine position is the control variable 
config.params["turbine_pos"] = [] 
border_x = 80.
border_y = 20.
for x_r in numpy.linspace(0.+border_x, config.params["basin_x"]-border_x, 2):
    for y_r in numpy.linspace(0.+border_y, config.params["basin_y"]-border_y, 2):
      config.params["turbine_pos"].append((float(x_r), float(y_r)))

info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
# Choosing a friction coefficient of > 0.25 ensures that overlapping turbines will lead to
# less power output.
config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))

model = ReducedFunctional(config, scaling_factor = -10**-6, plot = True)
m0 = model.initial_control()

#p = numpy.random.rand(len(m0))
#minconv = test_gradient_array(model.j, model.dj, m0, seed=0.1, perturbation_direction=p)
#if minconv < 1.9:
#    info_red("The gradient taylor remainder test failed.")
#  sys.exit(1)

g = lambda m: []
dg = lambda m: []

fmin_l_bfgs_b(model.j, m0, fprime = model.dj)
