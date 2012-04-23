''' This example optimises the position of three turbines using the hallow water model. '''

import configuration 
import numpy
from dirichlet_bc import DirichletBCSet
import IPOptUtils
import ipopt
from helpers import test_gradient_array
from animated_plot import *
from reduced_functional import ReducedFunctional
from dolfin import *
set_log_level(ERROR)

# An animated plot to visualise the development of the functional value
plot = AnimatedPlot(xlabel='Iteration', ylabel='Functional value')

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.ConstantInflowPeriodicSidesPaperConfiguration(nx=100, ny=33)

# The turbine position is the control variable 
turbine_pos = [[60, 38], [80, 28], [100, 38], [120, 28], [140, 38]] 

config.set_turbine_pos(turbine_pos)
info_blue("Deployed " + str(len(turbine_pos)) + " turbines.")

model = ReducedFunctional(config, scaling_factor = 10**-6, plot = True)
m0 = model.initial_control()

g = lambda m: []
dg = lambda m: []

f = IPOptUtils.IPOptFunction()
# Overwrite the functional and gradient function with our implementation
f.objective= model.j 
#f.gradient= model.dj_with_check 
f.gradient= model.dj

# Get the upper and lower bounds for the turbine positions
lb, ub = IPOptUtils.position_constraints(config.params)

nlp = ipopt.problem(len(m0), 
                    0, 
                    f, 
                    numpy.array(lb),
                    numpy.array(ub))
nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption('tol', 1e-9)
nlp.addOption('print_level', 5)
nlp.addOption('check_derivatives_for_naninf', 'yes')
# A -1.0 scaling factor transforms the min problem to a max problem.
nlp.addOption('obj_scaling_factor', -1.0)
# Use an approximate Hessian since we do not have second order information.
nlp.addOption('hessian_approximation', 'limited-memory')
nlp.addOption('max_iter', 50)

m, sinfo = nlp.solve(m0)
info(sinfo['status_msg'])
info("Solution of the primal variables: m=%s" % repr(m))
info("Solution of the dual variables: lambda=%s" % repr(sinfo['mult_g']))
info("Objective=%s" % repr(sinfo['obj_val']))

list_timings()
