import sys
import sw_config 
import sw_lib
import numpy
import Memoize
import IPOptUtils
import ipopt 
import numpy
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array
from turbines import *

# Global counter variable for vtk output
count = 0

def default_config():
  config = sw_config.DefaultConfiguration(nx=100, ny=25)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  config.params["finish_time"] = 2./4*period
  config.params["dt"] = config.params["finish_time"]/20
  print "Wave period (in h): ", period/60/60 
  config.params["dump_period"] = 1
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4 

  # Turbine settings
  config.params["friction"] = 0.0025
  config.params["turbine_pos"] = [[1000., 500.], [2000., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0
  config.params["turbine_length"] = 75
  config.params["turbine_width"] = 150

  # Now create the turbine measure
  config.initialise_turbines_measure()
  return config

def initial_control(config):
  return numpy.array([0.2])

def j_and_dj(x):
  adjointer.reset()
  adj_variables.__init__()
  
  set_log_level(30)
  debugging["record_all"] = True

  W=sw_lib.p1dgp2(config.mesh)

  # Get initial conditions
  state=Function(W)
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U)
  # Apply the control

  # Set up the turbine friction field using the provided control variable
  turbine_friction_orig = config.params["turbine_friction"]
  config.params["turbine_friction"] = x[0] * turbine_friction_orig
  tf.interpolate(GaussianTurbines(config))
  config.params["turbine_friction"] = turbine_friction_orig 

  global count
  count+=1
  sw_lib.save_to_file_scalar(tf, "turbines_"+str(count))

  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)
  def functional(state):
    turbines = GaussianTurbines(config)
    return config.params["dt"]*turbines*Constant(x[0])*0.5*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

  # Solve the shallow water system
  j, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)
  #print "Layout power extraction: ", j/1000000, " MW."
  #print "Which is equivalent to a average power generation of: ",  j/1000000/(config.params["current_time"]-config.params["start_time"]), " MW"

  #sw_lib.replay(state, config.params)

  J = TimeFunctional(functional(state))
  adj_state = sw_lib.adjoint(state, config.params, J, until=1)

  #sw_lib.save_to_file(adj_state, "adjoint")

  # we have dJ/dx = (\partial J)/(\partial turbine_friction) * (d turbine_friction) / d x +
  #                 + \partial J / \partial x
  #               = adj_state * turbine_friction
  #                 + \partial J / \partial x
  # In this particular case, j = \sum_t(functional) and \partial functional / \partial x = funtional/x. Hence we haev \partial J / \partial x = j/x
  tf.interpolate(GaussianTurbines(config))
  v = adj_state.vector()
  dj = v.inner(tf.vector()) 
  dj += j/x[0]

  return j, numpy.array([dj])

j_and_dj_mem = Memoize.MemoizeMutable(j_and_dj)
def j(x):
  j = j_and_dj_mem(x)[0]*10**-13
  print 'Evaluating j(', x[0].__repr__(), ')=', j
  return j 

def dj(x):
  dj = j_and_dj_mem(x)[1]*10**-13
  print 'Evaluating dj(', x[0].__repr__(), ')=', dj
  return dj

config = default_config()
x0 = initial_control(config)

p = numpy.array([1.])
#minconv = test_gradient_array(j, dj, x0, seed=0.0001, perturbation_direction=p)
#if minconv < 1.99:
#  print "The gradient taylor remainder test failed."
#  sys.exit(1)

# If this option does not produce any ipopt outputs, delete the ipopt.opt file
g = lambda x: []
dg = lambda x: []

f = IPOptUtils.IPOptFunction()
# Overwrite the functional and gradient function with our implementation
f.objective= j 
f.gradient= dj 

nlp = ipopt.problem(1, 0, f, numpy.array([0]), numpy.array([100])) 
#nlp.addOption('derivative_test', 'first-order')
nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption('tol', 1e-7)
nlp.addOption('print_level', 5)
# A -1.0 scaling factor transforms the min problem to a max problem
nlp.addOption('obj_scaling_factor', -1.0)
nlp.addOption('check_derivatives_for_naninf', 'yes')

x, info = nlp.solve(x0)
print info['status_msg'], " (status: ", info['status'], ")"
print "Solution of the primal variables: x=%s\n" % repr(x) 
print "Solution of the dual variables: lambda=%s\n" % repr(info['mult_g'])
print "Objective=%s\n" % repr(info['obj_val'])

if info['status'] != 0 or x[0] > 0: 
  print "The optimisation algorithm did not find the correct solution."
  sys.exit(1) 
else:
  exit_code = 0
