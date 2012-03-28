import sys
import sw_config 
import sw_lib
from dolfin import *
from dolfin_adjoint import *
from math import log

set_log_level(30)
myid = MPI.process_number()

def error(config):
  W=sw_lib.p1dgp2(config.mesh)
  initstate=Function(W)
  initstate.interpolate(config.get_sin_initial_condition()())
  u_exact = "eta0*sqrt(g*depth) * cos(k*x[0]-sqrt(g*depth)*k*t)" # The analytical veclocity of the shallow water equations has been multiplied by depth to account for the change of variable (\tilde u = depth u) in this code.
  du_exact = "(- eta0*sqrt(g*depth) * sin(k*x[0]-sqrt(g*depth)*k*t) * k)"
  eta_exact = "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"
  # The source term
  source = Expression(("friction*friction*g/(pow(depth, (1.0/3.0))) * " + u_exact, 
                       "0.0"), \
                       eta0=config.params["eta0"], g=config.params["g"], \
                       depth=config.params["depth"], t=config.params["current_time"], k=config.params["k"], friction = config.params["friction"])

  finalstate = sw_lib.sw_solve(W, config, initstate, annotate=False, u_source = source)

  analytic_sol = Expression((u_exact, \
                             "0", \
                             eta_exact), \
                             eta0=config.params["eta0"], g=config.params["g"], \
                             depth=config.params["depth"], t=config.params["current_time"], k=config.params["k"])
  exactstate=Function(W)
  exactstate.interpolate(analytic_sol)
  e = finalstate-exactstate
  return sqrt(assemble(dot(e,e)*dx))

def test(refinment_level):
  config = sw_config.DefaultConfiguration(nx=2*2**refinment_level, ny=2*2**refinment_level) 
  config.params["finish_time"] = pi/(sqrt(config.params["g"]*config.params["depth"])*config.params["k"])/10
  config.params["dt"] = config.params["finish_time"]/75
  config.params["dump_period"] = 100000
  config.params["friction"] = 0.0025 
  config.params["quadratic_friction"] = False

  return error(config)

errors = []
tests = 4
for refinment_level in range(1, tests):
  errors.append(test(refinment_level))
# Compute the order of convergence 
conv = [] 
for i in range(len(errors)-1):
  conv.append(abs(log(errors[i+1]/errors[i], 2)))

if myid == 0:
  print "Spatial order of convergence (expecting 2.0):", conv
if min(conv)<1.8:
  if myid == 0:
    print "Spatial convergence test failed for wave_flather"
  sys.exit(1)
else:
  if myid == 0:
    print "Test passed"

sys.exit()

adj_html("sw_forward.html", "forward")
adj_html("sw_adjoint.html", "adjoint")
sw_lib.replay(state, config.params)

J = Functional(dot(state, state)*dx)
f_direct = assemble(dot(state, state)*dx)
adj_state = sw_lib.adjoint(state, config.params, J)

ic = Function(W)
ic.interpolate(config.InitialConditions())
def J(ic):
  state = sw_lib.sw_solve(W, config, ic, annotate=False)
  analytic_sol = Expression("eta0*cos(pi/3000*x[0]-sqrt(g/h)*pi/3000*t")
  return assemble(dot(state, state)*dx)

minconv = test_initial_condition(J, ic, adj_state, seed=0.001)
if minconv < 1.9:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)
