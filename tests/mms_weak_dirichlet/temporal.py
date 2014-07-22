''' Tests the temporal order of convergence with weakly imposed
    Dirichlet conditions. '''

import sys
import math
from opentidalfarm import *
from opentidalfarm.initial_conditions import SinusoidalInitialCondition
from dolfin_adjoint import adj_reset
from dolfin import log, INFO, ERROR

def error(config, eta0, k):
  state = Function(config.function_space)
  state.interpolate(SinusoidalInitialCondition(config, eta0, k, 
      config.params["depth"]))

  adj_reset()
  shallow_water_model.sw_solve(config, state, annotate=False)

  analytic_sol = Expression(
         ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0", \
         "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"), \
         eta0=eta0, g=config.params["g"], \
         depth=config.params["depth"], \
         t=config.params["current_time"], k=k)
  exactstate = Function(config.function_space)
  exactstate.interpolate(analytic_sol)
  e = state - exactstate
  return sqrt(assemble(dot(e,e)*dx))

def test(refinement_level):
  nx = 2**4
  config = configuration.DefaultConfiguration(nx=nx, ny=2, 
          finite_element=finite_elements.p1dgp2) 
  domain = domains.RectangularDomain(3000, 1000, nx, 2)
  config.set_domain(domain)
  eta0 = 2.0
  k = pi/config.domain.basin_x
  config.params["finish_time"] = 2 * pi / (sqrt(config.params["g"] * 
      config.params["depth"]) * k)
  config.params["dt"] = config.params["finish_time"] / (4 * 2**refinement_level)
  config.params["theta"] = 0.5
  config.params["dump_period"] = 100000
  config.params["bctype"] = "dirichlet"
  config.params["u_weak_dirichlet_bc_expr"] = Expression(
         ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), 
         eta0=eta0, 
         g=config.params["g"], 
         depth=config.params["depth"], 
         t=config.params["current_time"], 
         k=k)

  return error(config, eta0, k)

errors = []
tests = 4
for refinement_level in range(tests):
  errors.append(test(refinement_level))
# Compute the order of convergence 
conv = [] 
for i in range(len(errors)-1):
  conv.append(abs(math.log(errors[i+1] / errors[i], 2)))

log(INFO, "Temporal order of convergence (expecting 2.0): %s" % str(conv))
if min(conv)<1.8:
  log(ERROR, "Temporal convergence test failed for wave_dirichlet")
  sys.exit(1)
else:
  log(INFO, "Test passed")
