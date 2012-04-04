import sys
import configuration 
import shallow_water_model as sw_model
import function_spaces
from initial_conditions import SinusoidalInitialCondition
from dolfin import *
from math import log

set_log_level(PROGRESS)
parameters["std_out_all_processes"] = False;

def error(config):
  W = function_spaces.p1dgp2(config.mesh)
  state = Function(W)
  state.interpolate(SinusoidalInitialCondition(config)())

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
  config = configuration.DefaultConfiguration(nx=2**8, ny=2) 
  config.params["finish_time"] = 2*pi/(sqrt(config.params["g"]*config.params["depth"])*config.params["k"])
  config.params["dt"] = config.params["finish_time"]/(2*2**refinment_level)
  config.params["theta"] = 0.5
  config.params["dump_period"] = 100000
  config.params["bctype"] = "dirichlet"

  class InitialConditions(Expression):
      def __init__(self):
          pass
      def eval(self, values, X):
          values[0]=config.params['eta0']*sqrt(config.params['g']*config.params['depth'])*cos(config.params["k"]*X[0])
          values[1]=0.
          values[2]=config.params['eta0']*cos(config.params["k"]*X[0])
      def value_shape(self):
          return (3,)

  config.InitialConditions = InitialConditions
  return error(config)

errors = []
tests = 6
for refinment_level in range(1, tests):
  errors.append(test(refinment_level))
# Compute the order of convergence 
conv = [] 
for i in range(len(errors)-1):
  conv.append(abs(log(errors[i+1]/errors[i], 2)))

info("Temporal order of convergence (expecting 2.0): %s" % str(conv))
if min(conv)<1.8:
  info_red("Temporal convergence test failed for wave_dirichlet")
  sys.exit(1)
else:
  info_green("Test passed")
