import sys
import opentidalfarm.domains
import math
from opentidalfarm import *
from opentidalfarm.initial_conditions import SinusoidalInitialCondition
from dolfin_adjoint import adj_reset 
from dolfin import log, INFO, ERROR

set_log_level(PROGRESS)
parameters["std_out_all_processes"] = False;

def error(config, eta0, k):
    state = Function(config.function_space)
    state.interpolate(SinusoidalInitialCondition(config, eta0, k, 
        config.params["depth"]))
    # The analytical veclocity of the shallow water equations has been multiplied 
    # by depth to account for the change of variable (\tilde u = depth u) 
    # in this code.
    u_exact = "eta0*sqrt(g/depth) * cos(k*x[0]-sqrt(g*depth)*k*t)"
    ddu_exact = "(viscosity * eta0*sqrt(g/depth) * cos(k*x[0]-sqrt(g*depth)*k*t) * k*k)"
    eta_exact = "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"
  
    # The source term
    source = Expression((ddu_exact, 
                        "0.0"),
                        eta0=eta0, g=config.params["g"],
                        depth=config.params["depth"], 
                        t=config.params["current_time"],
                        k=k, viscosity=config.params["viscosity"])
  
    adj_reset()
    shallow_water_model.sw_solve(config, state, annotate=False, u_source=source)
  
    analytic_sol = Expression((u_exact,
                               "0",
                               eta_exact),
                               eta0=eta0, g=config.params["g"],
                               depth=config.params["depth"], 
                               t=config.params["current_time"], 
                               k=k)
    exactstate = Function(config.function_space)
    exactstate.interpolate(analytic_sol)
    e = state - exactstate
    return sqrt(assemble(dot(e,e)*dx))

def test(refinement_level):
    nx = 16*2**refinement_level
    ny = 2**refinement_level
  
    config = configuration.DefaultConfiguration(nx=nx, ny=ny) 
    domain = opentidalfarm.domains.RectangularDomain(3000, 1000, nx, ny)
    config.set_domain(domain)
  
    eta0 = 2.0
    k = pi/config.domain.basin_x
    config.params["finish_time"] = pi/(sqrt(config.params["g"] * 
                                       config.params["depth"]) * k) / 20
    config.params["dt"] = config.params["finish_time"] / 100
    config.params["dump_period"] = 100000
    config.params["include_viscosity"] = True
    config.params["viscosity"] = 10.0
    config.params["flather_bc_expr"] = Expression(
        ("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"), 
        eta0=eta0, 
        g=config.params["g"], 
        depth=config.params["depth"], 
        t=config.params["current_time"], 
        k=k
    )
  
    return error(config, eta0, k)

errors = []
tests = 4
for refinement_level in range(1, tests):
    errors.append(test(refinement_level))
# Compute the order of convergence 
conv = [] 
for i in range(len(errors)-1):
    conv.append(abs(math.log(errors[i+1]/errors[i], 2)))

info_green("Errors: %s" % str(errors))
info_green("Spatial order of convergence (expecting 2.0): %s" % str(conv))
if min(conv)<1.8:
    log(ERROR, "Spatial convergence test failed for wave_flather")
    sys.exit(1)
else:
    log(INFO, "Test passed")
