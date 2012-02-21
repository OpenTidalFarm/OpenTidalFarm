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

M,G,rhs_contr,ufl = sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)

functional = DefaultFunctional(config.params) 
myj, djdm, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)

sw_lib.replay(state, config.params)

J = TimeFunctional(functional.Jt(state))
adj_state = sw_lib.adjoint(state, config.params, J)

ic = Function(W)
ic.interpolate(config.get_sin_initial_condition()())
def J(ic):
  j, djdm, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, ic, config.params, time_functional=functional, annotate=False)
  return j

minconv = test_initial_condition_adjoint(J, ic, adj_state, seed=0.0001)
if minconv < 1.9:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)
