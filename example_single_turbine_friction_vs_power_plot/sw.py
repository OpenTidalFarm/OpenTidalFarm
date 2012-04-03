''' An example that plots the power ouput of a single turbine for different fricition values. (see also test_optimal_friction_single_turbine) 
 - a single turbine (with a bump function as friction distribution)
 - shallow water model with implicit timestepping scheme to avoid oscillations in the turbine areas 
 - the functional is \int C * f * ||u||**3 where C is a constant
 '''

import sys
import sw_config 
import function_spaces
import numpy
import matplotlib.pyplot as plt
from dolfin import *
from dolfin_adjoint import *
from reduced_functional import ReducedFunctional

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=20, ny=10)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  # Start at rest state
  config.params["element_type"] = function_spaces.p2p1
  config.params["start_time"] = period/4 
  config.params["finish_time"] = period/2
  config.params["dt"] = config.params["finish_time"]/10
  info_green("Wave period (in h): %f" % (period/60/60, ))
  config.params["dump_period"] = 1
  config.params["verbose"] = 0
  # We need a implicit scheme to avoid oscillations in the turbine areas.
  config.params["theta"] = 1.0

  config.params["include_advection"] = True 
  config.params["include_diffusion"] = True 
  config.params["diffusion_coef"] = 20.0

  # Turbine settings
  config.params["friction"] = 0.0025
  config.params["turbine_pos"] = [[1500., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 600
  config.params["turbine_y"] = 600
  config.params["controls"] = ["turbine_friction"]
  # Solver options
  config.params["picard_iterations"] = 4

  return config

config = default_config()
model = ReducedFunctional(config)
m0 = model.initial_control()
m_list = [m0*i for i in numpy.linspace(0., 0.1, 15)]
info_green("Tested friction coefficients: " + str([fx[0] for fx in f]))

# Produce the power values for linear friction
info_green("Compute values for linear friction")
P = []
for m in m_list: 
  P.append(model.j(m, forward_only = True))

# Produce the power values for quadratic friction
info_green("Compute values for quadratic friction")
config.params["quadratic_friction"] = True 
model = ReducedFunctional(config)
P_quad = []
for m in m_list: 
  P_quad.append(model.j(m, forward_only = True))

# Plot the results
if MPI.process_number() == 0:
  plt.figure(1)
  plt.plot(f, P)
  info_green("Linear friction: The maximum functional value of " + str(max(P)) + " is achieved with a friction coefficient of " + str(f[numpy.argmax(P)]) + ".")
  plt.title('Power output of a single turbine with linear friction.')
  plt.ylabel('Power output')
  plt.xlabel('Linear friction coefficient')

  plt.figure(2)
  plt.plot(f, P_quad)
  info_green("Quadratic friction: The maximum functional value of " + str(max(P_quad)) + " is achieved with a friction coefficient of " + str(f[numpy.argmax(P_quad)]) + ".")
  plt.title('Power output of a single turbine with quadratic friction.')
  plt.ylabel('Power output')
  plt.xlabel('Quadratic friction coefficient')
  plt.show()
  plt.hold()
