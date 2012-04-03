import sys
import configuration 
import shallow_water_model as sw_model
import function_spaces
from dolfin import *
from dolfin_adjoint import *
from math import log

set_log_level(30)
myid = MPI.process_number()

def error(config):
  W = function_spaces.p1dgp2(config.mesh)
  state = Function(W)
  state.interpolate(config.get_sin_initial_condition()())

  sw_model.sw_solve(W, config, state, annotate=False)

  analytic_sol = Expression(("eta0*sqrt(g*depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", \
                             "0", \
                             "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"), \
                             eta0=config.params["eta0"], g=config.params["g"], \
                             depth=config.params["depth"], t=config.params["current_time"], k=config.params["k"])
  exactstate = Function(W)
  exactstate.interpolate(analytic_sol)
  e = state-exactstate
  return sqrt(assemble(dot(e,e)*dx))

def test(refinment_level):
  config = configuration.DefaultConfiguration(nx=2*2**refinment_level, ny=2) 
  config.params["finish_time"]=pi/(sqrt(config.params["g"]*config.params["depth"])*config.params["k"])/10
  config.params["dt"]=config.params["finish_time"]/100
  config.params["dump_period"]=100000
  config.params["bctype"]="dirichlet"

  return error(config)

errors = []
tests = 6
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
    print "Spatial convergence test failed for wave_dirichlet"
  sys.exit(1)
else:
  if myid == 0:
    print "Test passed"
