''' Tests the temporal order of convergence with strongly imposed
    Dirichlet conditions. '''

import sys
import math
from opentidalfarm import *
from opentidalfarm.initial_conditions import SinusoidalInitialCondition
from dolfin_adjoint import adj_reset
from dolfin import log, INFO, ERROR

def error(config, eta0, k):
  state = Function(config.function_space)
  ic_expr = SinusoidalInitialCondition(config, eta0, k, config.params["depth"])
  ic = project(ic_expr, state.function_space())
  state.assign(ic, annotate=False)

  adj_reset()
  shallow_water_model.sw_solve(config, state, annotate=False)

  analytic_sol = Expression(
         ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0", \
         "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"), \
         eta0=eta0, g=config.params["g"], \
         depth=config.params["depth"], \
         t=config.params["current_time"], k=k)
  return errornorm(analytic_sol, state)

def test(refinement_level):
    nx = 2**3
    ny = 1
    config = configuration.DefaultConfiguration(nx, ny)
    domain = domains.RectangularDomain(3000, 1000, nx, ny)
    config.set_domain(domain)
    eta0 = 2.0
    k = pi/config.domain.basin_x
    config.params["finish_time"] = 2 * pi / (sqrt(config.params["g"] * 
        config.params["depth"]) * k)
    config.params["dt"] = Constant(config.params["finish_time"] / 
            (4 * 2**refinement_level))
    config.params["theta"] = 0.5
    config.params["dump_period"] = 100000
    config.params["bctype"] = "strong_dirichlet"
    bc = DirichletBCSet(config)

    expression = Expression(
        ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), 
        eta0=eta0, 
        g=config.params["g"], 
        depth=config.params["depth"], 
        t=config.params["current_time"], 
        k=k)

    bc.add_analytic_u(1, expression)
    bc.add_analytic_u(2, expression)
    bc.add_analytic_u(3, expression)
    config.params["strong_bc"] = bc

    return error(config,eta0,k)

errors = []
tests = 4
for refinement_level in range(tests):
  errors.append(test(refinement_level))
# Compute the order of convergence 
conv = [] 
for i in range(len(errors)-1):
  conv.append(abs(math.log(errors[i+1] / errors[i], 2)))

log(INFO, "Temporal order of convergence (expecting 2.0): %s" % str(conv))
if min(conv) < 1.8:
  log(ERROR, "Temporal convergence test failed for wave_dirichlet")
  sys.exit(1)
else:
  log(INFO, "Test passed")
