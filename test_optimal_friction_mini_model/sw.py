''' Test description:
 - single turbine (with constant friction distribution) whose size exceeds the size of the domain
 - constant velocity profile with an initial x-velocity of 2.
 - control: turbine friction
 - the mini model will compute a x-velocity of 2/(f + 1) wher ef is the turbine friction.
 - the functional is \int C * f * ||u||**3 where C is a constant
 - hence we maximise C * f * ( 2/(f + 1) )**3, f > 0 which has the solution f = 0.5

 Note: The solution is known only because we use a constant turbine friction distribution. 
       However this turbine model is not differentiable at its boundary, and this is why
       the turbine size has to exceed the domain.
 '''

import sys
import sw_config 
import numpy
import ipopt 
import IPOptUtils
from sw_utils import test_gradient_array, pprint
from mini_model import *
from default_model import DefaultModel
from dolfin import *

# Global counter variable for vtk output
count = 0

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=20, ny=10)
  config.params["verbose"] = 0

  # dt is used in the functional only, so we set it here to 1.0
  config.params["dt"] = 1.0
  # Turbine settings
  config.params["turbine_pos"] = [[500., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 8000
  config.params["turbine_y"] = 8000
  config.params["turbine_model"] = "ConstantTurbine"

  return config

config = default_config()
model = DefaultModel(config, scaling_factor = 10**-5, forward_model = mini_model_solve)
m0 = model.initial_control()

p = numpy.random.rand(len(m0))
minconv = test_gradient_array(model.j, model.dj, m0, seed=0.001, perturbation_direction=p)
if minconv < 1.99:
  pprint("The gradient taylor remainder test failed.")
  sys.exit(1)

# If this option does not produce any ipopt outputs, delete the ipopt.opt file
g = lambda m: []
dg = lambda m: []

f = IPOptUtils.IPOptFunction()
# Overwrite the functional and gradient function with our implementation
f.objective = model.j 
f.gradient = model.dj 

nlp = ipopt.problem(len(m0), 
                    0, 
                    f, 
                    numpy.zeros(len(m0)), 
                    100*numpy.ones(len(m0)))
nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption('tol', 1e-9)
nlp.addOption('print_level', 5)
nlp.addOption('check_derivatives_for_naninf', 'yes')
# A -1.0 scaling factor transforms the min problem to a max problem.
nlp.addOption('obj_scaling_factor', -1.0)
# Use an approximate Hessian since we do not have second order information.
nlp.addOption('hessian_approximation', 'limited-memory')
nlp.addOption('max_iter', 7)

m, info = nlp.solve(m0)
pprint(info['status_msg'])
pprint("Solution of the primal variables: m=%s\n" % repr(m))
pprint("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))
pprint("Objective=%s\n" % repr(info['obj_val']))

if info['status'] != 0 or abs(m[0]-0.5) > 10**-10: 
  pprint("The optimisation algorithm did not find the correct solution.")
  sys.exit(1) 
