import sys
import sw_config 
import sw_lib
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array
from turbines import *

def default_config():
  config = sw_config.DefaultConfiguration(nx=20, ny=5)
  period = 1.24*60*60 # Wave period
  config.params["k"]=2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  config.params["finish_time"]=2./4*period
  config.params["dt"]=config.params["finish_time"]/40
  print "Wave period (in h): ", period/60/60 
  config.params["dump_period"]=1
  config.params["verbose"] = 0

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
  return config

def initial_control(config):
  W=sw_lib.p1dgp2(config.mesh)

  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U)
  tf.interpolate(GaussianTurbines(config))
  return tf.vector().array()

def j(x):
  return j_and_dj(x)[0]

def dj(x):
  return j_and_dj(x)[1]

def j_and_dj(x):
  adj_html("sw_forward.html", "forward")
  #adjointer.__init__()
  adjointer.reset()
  adj_variables.__init__()
  
  set_log_level(30)
  debugging["record_all"] = True

  W=sw_lib.p1dgp2(config.mesh)

  state=Function(W)
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U)
  tf.vector().set_local(x) 
  sw_lib.save_to_file_scalar(tf, "turbines")

  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)
  def functional(state):
    turbines = GaussianTurbines(config)
    #plot(turbines/12*50*(dot(state[0], state[0]) + dot(state[1], state[1])))
    #return config.params["dt"]*0.5*config.params["turbine_friction"]*(dot(state[0], state[0]) + dot(state[1], state[1])/(config.params["g"]*config.params["depth"]))**1.5*config.dx(1)
    return config.params["dt"]*0.5*turbines*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

  adj_html("sw_forward1.html", "forward")
  j, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)
  print "Layout power extraction: ", j/1000000, " MW."
  print "Which is equivalent to a average power generation of: ",  j/1000000/(config.params["current_time"]-config.params["start_time"]), " MW"

  #sw_lib.replay(state, config.params)

  J = TimeFunctional(functional(state))
  adj_state = sw_lib.adjoint(state, config.params, J, until=1)

  #sw_lib.save_to_file(adj_state, "adjoint")

  return j, adj_state.vector().array()

# main 
from openopt import NLP
config = default_config()
x0 = initial_control(config)

test_gradient_array(j, dj, x0, seed=0.0001)

p = NLP(j, x0, df=dj,  # c=c,  dc=dc, h=h,  dh=dh,  A=A,  b=b,  Aeq=Aeq,  beq=beq, 
    #lb=lb, ub=ub, gtol=gtol, contol=contol, iprint = 50, maxIter = 10000, 
    maxFunEvals = 1e7, name = 'NLP_1')
p.plot = True
p.goal ='max'
solver = 'ralg'
r = p.solve(solver, plot=0)
