''' This example optimises the position of three turbines using the hallow water model. '''

import sys
import cProfile
import pstats
import sw_config 
import sw_lib
import numpy
import Memoize
import IPOptUtils
import cProfile
from animated_plot import *
from functionals import DefaultFunctional
from sw_utils import test_initial_condition_adjoint, test_gradient_array, pprint
from turbines import *
from mini_model import *
from dolfin import *
from dolfin_adjoint import *

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  config = sw_config.DefaultConfiguration(nx=600, ny=200)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  config.params["finish_time"] = 2./4*period
  config.params["dt"] = config.params["finish_time"]/20
  pprint("Wave period (in h): ", period/60/60)
  config.params["dump_period"] = 1
  config.params["verbose"] = 100

  # Start at rest state
  config.params["start_time"] = config.params["finish_time"] - 3*config.params["dt"] #period/4 

  # Turbine settings
  config.params["friction"] = 0.0025
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [] 
  border = 100
  for x_r in numpy.linspace(0.+border, config.params["basin_x"]-border, 30):
    for y_r in numpy.linspace(0.+border, config.params["basin_y"]-border, 10):
      config.params["turbine_pos"].append((float(x_r), float(y_r)))

  info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
  # Choosing a friction coefficient of 1.0 ensures that overlapping turbines will lead to
  # less power output.
  config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 190 # We overlap the turbines on purpose
  config.params["turbine_y"] = 20

  return config


config = default_config()

W=sw_lib.p1dgp2(config.mesh)
state=Function(W)

# Set the control values
U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
U = U.collapse() # Recompute the DOF map
tf = Function(U) # The turbine function

# Set up the turbine friction field using the provided control variable
turbines = Turbines(config.params)
cProfile.run("tf.interpolate(turbines)")

print "norm(tf) = ", norm(tf)
correct_norm = 605.429678289 
if abs(norm(tf) - correct_norm) > 0.000000001:
  print "Warning: Wrong norm. Should be ", correct_norm 
