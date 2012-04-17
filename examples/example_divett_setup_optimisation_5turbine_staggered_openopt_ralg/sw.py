''' This example optimises the position of three turbines using the hallow water model. '''

import sys
import configuration 
import numpy
import IPOptUtils
from dirichlet_bc import DirichletBCSet
from helpers import test_gradient_array
from animated_plot import *
from reduced_functional import ReducedFunctional
from dolfin import *
from openopt import NLP
set_log_level(ERROR)

# An animated plot to visualise the development of the functional value
plot = AnimatedPlot(xlabel='Iteration', ylabel='Functional value')

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.ConstantInflowPeriodicSidesPaperConfiguration(nx=100, ny=33)

# The turbine position is the control variable 
config.params["turbine_pos"] = [[60, 38], [80, 28], [100, 38], [120, 28], [140, 38]] 

info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
# Choosing a friction coefficient of > 0.25 ensures that overlapping turbines will lead to
# less power output.
config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))

model = ReducedFunctional(config, scaling_factor = 10**-6, plot = True)
m0 = model.initial_control()

g = lambda m: []
dg = lambda m: []

# Get the upper and lower bounds for the turbine positions
lb, ub = IPOptUtils.position_constraints(config.params)

p = NLP(f = model.j, df = model.dj, x0 = m0, lb = lb, ub = ub, iprint = 1)
p.goal = 'max' 
r = p.solve('ralg')
print r.xf
