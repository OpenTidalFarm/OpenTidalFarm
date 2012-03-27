'''This tests checks the corrections of the adjoint by using it to compute the 
   derivative of the functional with respect to the initial condition.'''

import sys
import sw_config 
import sw_lib
import numpy
from turbines import *
from functionals import DefaultFunctional
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint

set_log_level(30)
myid = MPI.process_number()
debugging["record_all"] = True

config = sw_config.DefaultConfiguration(nx=10, ny=2) 
period = 1.24*60*60 # Wave period
config.params["k"]=2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
config.params["start_time"]=0
config.params["finish_time"]=period/10
config.params["dt"]=config.params["finish_time"]/10
if myid == 0:
  print "Wave period (in h): ", period/60/60 
config.params["dump_period"]=100000

# Turbine settings
config.params["friction"]=0.0025
config.params["turbine_pos"]=[[200., 500.], [1000., 700.]]
config.params["turbine_friction"] = 12.*numpy.ones(len(config.params["turbine_pos"]))
config.params["turbine_x"] = 400
config.params["turbine_y"] = 400
config.params["turbine_model"] = 'ConstantTurbine'

W=sw_lib.p1dgp2(config.mesh)

state=Function(W)
state.interpolate(config.get_sin_initial_condition()())

# Extract the first dimension of the velocity function space 
U = W.split()[0].sub(0)
U = U.collapse() # Recompute the DOF map
tf = Function(U)
tf.interpolate(Turbines(config.params))

def build_turbine_cache(params, function_space, turbine_size_scaling = 1.0):
  ''' Interpolates all the fields that are required for the Functional.'''
  turbine_cache = {}

  params = sw_lib.parameters(dict(params))
  # Scale the turbine size by the given factor.
  if turbine_size_scaling != 1.0:
    info_green("The functional uses turbines which size is scaled by a factor of " + str(turbine_size_scaling) + ".")
  params["turbine_x"] *= turbine_size_scaling
  params["turbine_y"] *= turbine_size_scaling

  turbines = Turbines(params)
  tf = Function(function_space)
  tf.interpolate(turbines)
  turbine_cache["turbine_field"] = tf

  # The derivatives with respect to the friction
  turbine_cache["turbine_derivative_friction"] = []
  for n in range(len(params["turbine_friction"])):
    tf = Function(U)
    turbines = Turbines(params, derivative_index_selector=n, derivative_var_selector='turbine_friction')
    tf = Function(function_space)
    tf.interpolate(turbines)
    turbine_cache["turbine_derivative_friction"].append(tf)

  # The derivatives with respect to the turbine position
  turbine_cache["turbine_derivative_pos"] = []
  for n in range(len(params["turbine_pos"])):
    turbine_cache["turbine_derivative_pos"].append({})
    for var in ('turbine_pos_x', 'turbine_pos_y'):
      tf = Function(U)
      turbines = Turbines(params, derivative_index_selector=n, derivative_var_selector=var)
      tf = Function(function_space)
      tf.interpolate(turbines)
      turbine_cache["turbine_derivative_pos"][-1][var] = tf

  return turbine_cache

turbine_cache = build_turbine_cache(config.params, U)
functional = DefaultFunctional(config.params, turbine_cache) 
myj, djdm, state = sw_lib.sw_solve(W, config, state, turbine_cache, time_functional=functional)

adj_html('annot.html', 'forward')
sw_lib.replay(state, config.params)

J = TimeFunctional(functional.Jt(state))
# Because no turbine field is used, the first equation in the annotation is the initialisation
# of the initial condition, hence the adjoint is computed all the way back to equation 0.
adj_state = sw_lib.adjoint(state, config.params, J, until = 2)

ic = Function(W)
ic.interpolate(config.get_sin_initial_condition()())
def J(ic):
  j, djdm, state = sw_lib.sw_solve(W, config, ic, turbine_cache, time_functional=functional, annotate=False)
  return j

minconv = test_initial_condition_adjoint(J, ic, adj_state, seed=0.0001)
if minconv < 1.9:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)
