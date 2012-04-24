''' This example optimises in an L shaped domain. '''
import configuration 
import numpy
from dirichlet_bc import DirichletBCSet
import IPOptUtils
import ipopt
from helpers import test_gradient_array
from animated_plot import *
from reduced_functional import ReducedFunctional
from domains import LShapeDomain
from dolfin import *
set_log_level(ERROR)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.ConstantInflowPeriodicSidesPaperConfiguration()

L_len = 200
config.set_domain( LShapeDomain("mesh.xml", L_len) )
# We also need to reapply the bc
bc = DirichletBCSet(config)
bc.add_constant_flow(1)
bc.add_noslip_u(3)
config.params['strong_bc'] = bc

# The turbine position is the control variable 
config.params["turbine_pos"] = [[60, 38], [80, 28], [100, 38], [120, 28], [140, 38]] 

info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
# Choosing a friction coefficient of > 0.25 ensures that overlapping turbines will lead to
# less power output.
config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))

model = ReducedFunctional(config, scaling_factor = 10**1, plot = True)
m0 = model.initial_control()

g = lambda m: []
dg = lambda m: []

f = IPOptUtils.IPOptFunction()
# Overwrite the functional and gradient function with our implementation
f.objective= model.j 
#f.gradient= model.dj_with_check 
f.gradient= model.dj

nlp = ipopt.problem(len(m0), 
                    0, 
                    f) 
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
