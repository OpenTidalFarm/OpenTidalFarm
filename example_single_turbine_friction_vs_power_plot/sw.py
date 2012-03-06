''' An example that plots the power ouput of a single turbine for different fricition values. 
 - a single turbine (with a bump function as friction distribution)
 - shallow water model with implicit timestepping scheme to avoid oscillations in the turbine areas 
 - the functional is \int C * f * ||u||**3 where C is a constant
 '''

import sys
import sw_config 
import sw_lib
import numpy
import Memoize
import ipopt 
import IPOptUtils
import matplotlib.pyplot as plt
from functionals import DefaultFunctional
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array, pprint
from turbines import *

# Global counter variable for vtk output
count = 0

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=20, ny=10)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  config.params["finish_time"] = period/2
  config.params["dt"] = config.params["finish_time"]/10
  pprint("Wave period (in h): ", period/60/60)
  config.params["dump_period"] = 1
  config.params["verbose"] = 0
  # We need a implicit scheme to avoid oscillations in the turbine areas.
  config.params["theta"] = 1.0

  # Start at rest state
  config.params["start_time"] = period/4 

  # Turbine settings
  config.params["friction"] = 0.0025
  config.params["turbine_pos"] = [[1500., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0*numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 600
  config.params["turbine_y"] = 600
  #config.params["bctype"]="dirichlet"

  return config

def initial_control(config):
  # We use the current turbine settings as the intial control
  res = config.params['turbine_friction'].tolist()
  return numpy.array(res)

def j(m):
  adjointer.reset()
  adj_variables.__init__()

  # Change the control variables to the config parameters
  config.params["turbine_friction"] = m[:len(config.params["turbine_friction"])]
  mp = m[len(config.params["turbine_friction"]):]

  set_log_level(30)
  debugging["record_all"] = True

  W=sw_lib.p1dgp2(config.mesh)

  # Set initial conditions
  state=Function(W)
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U) # The turbine function
  tfd = Function(U) # The derivative turbine function

  # Set up the turbine friction field using the provided control variable
  tf.interpolate(Turbines(config.params))

  global count
  count+=1
  sw_lib.save_to_file_scalar(tf, "turbines_t=."+str(count)+".x")

  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)

  functional = DefaultFunctional(config.params)

  # Solve the shallow water system
  j, djdm, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)

  return j

config = default_config()
m0 = initial_control(config)
f = [0.001*m0*i**2 for i in range(15)]
P = []
for fr in f: 
  P.append(j(fr))

if MPI.process_number() == 0:
  plt.plot(f, P)
  plt.title('Power output of a single turbine for different friction values.')
  plt.xlabel('Power output')
  plt.ylabel('Friction coefficient')
  plt.show()
  plt.hold()
