import sys
from opentidalfarm import *
from opentidalfarm.initial_conditions import SinusoidalInitialCondition
import opentidalfarm.domains
from dolfin_adjoint import adj_reset
from math import log

set_log_level(INFO)
parameters["std_out_all_processes"] = False;

def error(config, eta0, k):
  state = Function(config.function_space)
  state.interpolate(SinusoidalInitialCondition(config, eta0, k, config.params["depth"]))
  u_exact = "eta0*sqrt(g/depth) * cos(k*x[0]-sqrt(g*depth)*k*t)" 
  du_exact = "(- eta0*sqrt(g/depth) * sin(k*x[0]-sqrt(g*depth)*k*t) * k)"
  eta_exact = "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"
  # The source term
  source = Expression(("friction/depth * " + u_exact + "*pow(pow(" + u_exact + ", 2), 0.5)",  
                       "0.0"), \
                       eta0=eta0, g=config.params["g"], \
                       depth=config.params["depth"], t=config.params["current_time"], k=k, friction = config.params["friction"])

  adj_reset()
  shallow_water_model.sw_solve(config, state, annotate=False, u_source = source)

  analytic_sol = Expression((u_exact, \
                             "0", \
                             eta_exact), \
                             eta0=eta0, g=config.params["g"], \
                             depth=config.params["depth"], t=config.params["current_time"], k=k)
  exactstate = Function(config.function_space)
  exactstate.interpolate(analytic_sol)
  e = state - exactstate
  return sqrt(assemble(dot(e,e)*dx))

def test(refinement_level):
  config = configuration.DefaultConfiguration(nx=2*2**refinement_level, ny=2*2**refinement_level, finite_element = finite_elements.p1dgp2) 
  config.set_domain(opentidalfarm.domains.RectangularDomain(3000, 1000, 2*2**refinement_level, 2*2**refinement_level))
  eta0 = 2.0
  k = pi/config.domain.basin_x
  config.params['k'] = k
  config.params["finish_time"] = pi/(sqrt(config.params["g"]*config.params["depth"])*k)/10
  config.params["dt"] = config.params["finish_time"]/150
  config.params["dump_period"] = 100000
  config.params["friction"] = 0.25 
  config.params["quadratic_friction"] = True
  config.params["newton_solver"] = True

  return error(config, eta0, k)

errors = []
tests = 3
for refinement_level in range(1, tests):
  errors.append(test(refinement_level))
# Compute the order of convergence 
conv = [] 
for i in range(len(errors)-1):
  conv.append(abs(log(errors[i+1]/errors[i], 2)))

info("Spatial order of convergence (expecting 2.0): %s" % str(conv))
if min(conv)<1.8:
  info_red("Spatial convergence test failed for wave_flather")
  sys.exit(1)
else:
  info_green("Test passed")
