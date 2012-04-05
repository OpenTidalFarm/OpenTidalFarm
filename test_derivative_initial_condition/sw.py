'''This tests checks the corrections of the adjoint by using it to compute the 
   derivative of the functional with respect to the initial condition.'''

import sys
import configuration 
import shallow_water_model as sw_model
import finite_elements
import numpy
from initial_conditions import SinusoidalInitialCondition
from turbines import *
from functionals import DefaultFunctional
from dolfin import *
from dolfin_adjoint import *

set_log_level(PROGRESS)
myid = MPI.process_number()
debugging["record_all"] = True

config = configuration.DefaultConfiguration(nx=10, ny=2, finite_element = finite_elements.p1dgp2) 
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
config.params["functional_turbine_scaling"] = 1.0

# Setup the model and run it so that the annotation exists.
state = Function(config.function_space, name="Current_state")
state.interpolate(SinusoidalInitialCondition(config)())

functional = DefaultFunctional(config.function_space, config.params) 
sw_model.sw_solve(config.function_space, config, state, time_functional=functional)

# Check the replay
info("Replaying the forward model")
replay_dolfin()

# Run the adjoint model 
J = TimeFunctional(functional.Jt(state), static_variables = [functional.turbine_cache["turbine_field"]], dt=config.params["dt"])
dJdm = compute_gradient(J, InitialConditionParameter("Current_state"))

# And finally check the computed gradient with the taylor test
def J(state):
  j, djdm = sw_model.sw_solve(config.function_space, config, state, time_functional=functional, annotate=False)
  return j

state.interpolate(SinusoidalInitialCondition(config)())
minconv = test_initial_condition_adjoint(J, state, dJdm, seed=0.0001)
if minconv < 1.9:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)
