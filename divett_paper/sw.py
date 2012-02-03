import sys
import sw_config 
import sw_lib
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint
from turbines import *

set_log_level(30)
debugging["record_all"] = True

config = sw_config.SWConfiguration(nx=200, ny=50)
period = 1.24*60*60 # Wave period
config.params["k"]=2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
config.params["basename"]="p1dgp2"
config.params["finish_time"]=2./4*period
config.params["dt"]=config.params["finish_time"]/40
print "Wave period (in h): ", period/60/60 
config.params["dump_period"]=1
config.params["bctype"]="flather"

# Start at rest state
config.params["start_time"] = period/4 

# Turbine settings
config.params["friction"]=0.0025
config.params["turbine_pos"] = [[1000., 500.], [2000., 500.]]
config.params["turbine_friction"] = 12./config.params["depth"]
config.params["turbine_length"] = 200
config.params["turbine_width"] = 400

# Now create the turbine measure
config.initialise_turbines_measure()

############# Initial Conditions ##################
class InitialConditions(Expression):
    def __init__(self):
        pass
    def eval(self, values, X):
        values[0]=config.params['eta0']*sqrt(config.params['g']*config.params['depth'])*cos(config.params["k"]*X[0]-sqrt(config.params["g"]*config.params["depth"])*config.params["k"]*config.params["start_time"])
        values[1]=0.
        values[2]=config.params['eta0']*cos(config.params["k"]*X[0]-sqrt(config.params["g"]*config.params["depth"])*config.params["k"]*config.params["start_time"])
    def value_shape(self):
        return (3,)

W=sw_lib.p1dgp2(config.mesh)

state=Function(W)
state.interpolate(InitialConditions())

U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
U = U.collapse() # Recompute the DOF map
tf = Function(U)
tf.interpolate(GaussianTurbines(config))
sw_lib.save_to_file_scalar(tf, "turbines")

M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)
def functional(state):
  #turbines = GaussianTurbines(config)[0]
  #plot(turbines/12*50*(dot(state[0], state[0]) + dot(state[1], state[1])))
  return config.params["dt"]*0.5*config.params["turbine_friction"]*(dot(state[0], state[0]) + dot(state[1], state[1])/(config.params["g"]*config.params["depth"]))**1.5*config.dx(1)
  #return config.params["dt"]*0.5*turbines*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

initj, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)
print "Initial layout power extraction: ", initj/1000000, " MW."
print "Which is equivalent to a average power generation of: ",  initj/1000000/(config.params["current_time"]-config.params["start_time"]), " MW"
sys.exit()

adj_html("sw_forward.html", "forward")
adj_html("sw_adjoint.html", "adjoint")
sw_lib.replay(state, config.params)

J = TimeFunctional(functional(state))
adj_state = sw_lib.adjoint(state, config.params, J, until=1)

sw_lib.save_to_file(adj_state, "adjoint")

def J(tf):
  ic = Function(W)
  ic.interpolate(InitialConditions())
  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf[0])
  j, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, ic, config.params, time_functional=functional, annotate=False)
  return j

minconv = test_initial_condition_adjoint(J, tf, adj_state, seed=0.00001)
if minconv < 1.9:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)
