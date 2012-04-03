''' This test checks the performance of the turbine model implementation. One of the slowest part of the code is the interpolation of the
    turbine field onto a discrete function space, because turbine's eval function is called very often. This test was used to optimise the eval
    implementation. On 4 Intel(R) Xeon(R) CPU  E5506  @ 2.13GHz the benchmark time should be around 11s. ''' 

import sys
import configuration 
import sw_lib
import numpy
import cProfile
from functionals import DefaultFunctional
from turbines import *
from dolfin import *
from dolfin_adjoint import *

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  config = configuration.DefaultConfiguration(nx=600, ny=200)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  config.params["finish_time"] = 2./4*period
  config.params["dt"] = config.params["finish_time"]/20
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

W = function_spaces.p1dgp2(config.mesh)
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
