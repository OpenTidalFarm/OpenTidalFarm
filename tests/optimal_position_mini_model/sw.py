''' Test description
 - single turbine
 - bubble velocity profile with maximum in the center of the domain
 - control: turbine position
 - the optimal placement for the turbine is where the velocity profile reaches its maximum (the center of the domain)
'''

import sys
import configuration 
import numpy
import IPOptUtils
import finite_elements
from animated_plot import *
from helpers import test_gradient_array
from mini_model import mini_model_solve
from reduced_functional import ReducedFunctional
from initial_conditions import BumpInitialCondition
from dolfin import *
from scipy.optimize import fmin_slsqp

# An animated plot to visualise the development of the functional value
plot = AnimatedPlot(xlabel='Iteration', ylabel='Functional value')

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = configuration.DefaultConfiguration(nx=40, ny=20, finite_element = finite_elements.p1dgp2)
  config.params["verbose"] = 0

  # dt is used in the functional only, so we set it here to 0.8
  config.params["dt"] = 0.8
  # Turbine settings
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [[500., 200.]]
  config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 800
  config.params["turbine_y"] = 800
  config.params["controls"] = ['turbine_pos']
  config.params["functional_turbine_scaling"] = 1.0
  config.params["initial_condition"] = BumpInitialCondition

  return config

config = default_config()
model = ReducedFunctional(config, scaling_factor = -10**1, forward_model = mini_model_solve, plot = True)
m0 = model.initial_control()

p = numpy.random.rand(len(m0))
minconv = test_gradient_array(model.j, model.dj, m0, seed=0.001, perturbation_direction=p)
if minconv < 1.9:
  info_red("The gradient taylor remainder test failed.")
  sys.exit(1)

# If this option does not produce any ipopt outputs, delete the ipopt.opt file
g = lambda m: []
dg = lambda m: []

bounds = [(0, 3000), (0, 1000)] 

m = fmin_slsqp(model.j, m0, fprime = model.dj, bounds = bounds, iprint = 2)

info("Solution of the primal variables: m=" + repr(m) + "\n")
plot.savefig("plot_functional_value.png")

exit_code = 1
if abs(m[0]-1500) > 40:
    info_red("The optimisation algorithm did not find the optimal x position: %f instead of 1500." % m[0])
elif abs(m[1]-500) > 0.4:
    info_red("The optimisation algorithm did not find the optimal y position: %f instead of 500." %m[1])
else:
    info_green("Test passed")
    exit_code = 0

sys.exit(exit_code) 
