''' Test description
 - single turbine
 - bubble velocity profile with maximum in the center of the domain
 - control: turbine position
 - the optimal placement for the turbine is where the velocity profile reaches its maximum (the center of the domain)
'''

import sys
import cProfile
import pstats
import sw_config 
import sw_lib
import numpy
import memoize
import ipopt 
import IPOptUtils
from animated_plot import *
from sw_utils import test_gradient_array
from mini_model import mini_model_solve
from reduced_functional import ReducedFunctional
from dolfin import *
from dolfin_adjoint import *


# Global counter variable for vtk output
count = 0
# An animated plot to visualise the development of the functional value
plot = AnimatedPlot(xlabel='Iteration', ylabel='Functional value')

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=40, ny=20)
  config.params["verbose"] = 0

  # dt is used in the functional only, so we set it here to 1.0
  config.params["dt"] = 0.8
  # Turbine settings
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [[500., 200.]]
  config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 800
  config.params["turbine_y"] = 800
  config.params["controls"] = ['turbine_pos']
  config.params["functional_turbine_scaling"] = 1.0

  return config

class BumpInitialConditions(Expression):
  '''This class implements a initial condition with a bump velocity profile.
     With that we know that the optimal turbine location must be in the center of the domain. '''
  def __init__(self, config):
    self.params = config.params

  def bump_function(self, x):
    '''The velocity is initially a bump function (a smooth function with limited support):
               /  e**-1/(1-x**2)   for |x| < 1
      psi(x) = |  
               \  0   otherwise
      For more information see http://en.wikipedia.org/wiki/Bump_function
    '''
    bump = exp(-1.0/(1.0-x[0]**2)) 
    bump *= exp(-1.0/(1.0-x[1]**2)) 
    bump /= exp(-1)**2
    return bump

  def eval(self, values, X):
    params = self.params
    x_unit = 2*(params["basin_x"]-X[0])/params["basin_x"]-1.0
    y_unit = 2*(params["basin_y"]-X[1])/params["basin_y"]-1.0

    values[0] = self.bump_function([x_unit, y_unit]) 
    values[1] = 0.
    values[2] = 0.0 
  def value_shape(self):
    return (3,)

config = default_config()
initial_condition = BumpInitialConditions(config)
model = ReducedFunctional(config, scaling_factor = 10**4, forward_model = mini_model_solve, initial_condition = initial_condition)
m0 = model.initial_control()

p = numpy.random.rand(len(m0))
minconv = test_gradient_array(model.j, model.dj, m0, seed=0.001, perturbation_direction=p)
if minconv < 1.9:
  print "The gradient taylor remainder test failed."
  sys.exit(1)

# If this option does not produce any ipopt outputs, delete the ipopt.opt file
g = lambda m: []
dg = lambda m: []

f = IPOptUtils.IPOptFunction()
# Overwrite the functional and gradient function with our implementation
f.objective= model.j 
f.gradient= model.dj 

nlp = ipopt.problem(len(m0), 
                  0, 
                  f, 
                  numpy.zeros(len(m0)), 
                  3000.*numpy.ones(len(m0)))
nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption('tol', 1e-9)
nlp.addOption('print_level', 5)
nlp.addOption('check_derivatives_for_naninf', 'yes')
# A -1.0 scaling factor transforms the min problem to a max problem.
nlp.addOption('obj_scaling_factor', -1.0)
# Use an approximate Hessian since we do not have second order information.
nlp.addOption('hessian_approximation', 'limited-memory')
nlp.addOption('max_iter', 25)

m, info = nlp.solve(m0)
info_green(info['status_msg'])
info_green("Solution of the primal variables: m=" + repr(m) + "\n")
info_green("Solution of the dual variables: lambda=" +  repr(info['mult_g']) + "\n")
info_green("Objective=" + repr(info['obj_val']) + "\n")
plot.savefig("plot_functional_value.png")

exit_code = 1
if info['status'] != 0: 
    print "The optimisation algorithm did not find a solution."
elif abs(m[0]-1500) > 40:
    print "The optimisation algorithm did not find the optimal x position:", m[0] , "instead of 1500."
elif abs(m[1]-500) > 0.4:
    print "The optimisation algorithm did not find the optimal y position:", m[1] , "instead of 500."
else:
    exit_code = 0

sys.exit(exit_code) 
