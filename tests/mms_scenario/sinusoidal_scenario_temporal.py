import sys
from opentidalfarm import *
from dolfin_adjoint import adj_reset
from convergence_plot import *
import math 

set_log_level(ERROR)
parameters["std_out_all_processes"] = False;

def error(config):
  state = Function(config.function_space)
  state.interpolate(SinusoidalInitialCondition(config)())
  u_exact = "eta0*sqrt(g/depth) * cos(k*x[0]-sqrt(g*depth)*k*t)" 
  du_exact = "(- eta0*sqrt(g/depth) * sin(k*x[0]-sqrt(g*depth)*k*t) * k)"
  ddu_exact = "(diffusion_coef * eta0*sqrt(g/depth) * cos(k*x[0]-sqrt(g*depth)*k*t) * k*k)"
  eta_exact = "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"
  friction = "friction/depth * " + u_exact + "*pow(pow(" + u_exact + ", 2), 0.5)"
  # The source term
  source = Expression((u_exact + " * " + du_exact + " + " + ddu_exact + " + " + friction, 
                             "0.0"), \
                             eta0 = config.params["eta0"], g = config.params["g"], \
                             depth = config.params["depth"], t = config.params["current_time"], \
                             k = config.params["k"],  
                             diffusion_coef = config.params["diffusion_coef"],
                             friction = config.params["friction"])

  adj_reset()
  shallow_water_model.sw_solve(config, state, annotate=False, u_source = source)

  analytic_sol = Expression((u_exact, \
                             "0", \
                             eta_exact), \
                             eta0=config.params["eta0"], g=config.params["g"], \
                             depth=config.params["depth"], t=config.params["current_time"], k=config.params["k"])
  exactstate = Function(config.function_space)
  exactstate.interpolate(analytic_sol)
  e = state - exactstate
  enorm = sqrt(assemble(dot(e,e)*dx))
  print enorm
  return enorm 

def test(refinement_level):
  # Set up the configuration
  config = configuration.SinusoidalScenarioConfiguration("mesh_dummy.xml", [1, 0]) 
  config.set_domain( RectangularDomain(640., 320., nx = 2**8, ny = 2), warning = False)

  # Choose more appropriate timing settings for the mms test
  config.params["k"] = pi/config.domain.basin_x
  config.params["start_time"] = 0
  config.params["finish_time"] = pi/(sqrt(config.params["g"]*config.params["depth"])*config.params["k"])
  config.params["dt"] = config.params["finish_time"]/(2*2**refinement_level)

  # Make sure that we apply the analytical boundary conditions
  config.params['initial_condition'] = SinusoidalInitialCondition 
  config.params["dump_period"] = 100000
  config.params["eta0"] = 2.
  config.params["bctype"] = "strong_dirichlet"
  bc = DirichletBCSet(config)
  bc.add_analytic_u(1)
  bc.add_analytic_u(2)
  bc.add_analytic_u(3)
  config.params['strong_bc'] = bc

  return config.params["dt"], error(config)

errors = []
dts = []
tests = 7
for refinement_level in range(3, tests):
  dt, e = test(refinement_level)
  errors.append(e)
  dts.append(dt)
# Compute the order of convergence 
conv = [] 
for i in range(len(errors)-1):
  conv.append(abs(math.log(errors[i+1]/errors[i], 2)))

# Plot the result
save_convergence_plot(errors, dts, "Temporal rate of convergence", "Temporal error", order = 1.0, show_title = False, xlabel = "Time step [s]")

info_green("Errors: %s.", str(errors))
info_green("Temporal order of convergence (expecting 1.0): %s" % str(conv))
if min(conv)<0.88:
  info_red("Temporal convergence test failed for wave_flather")
  sys.exit(1)
else:
  info_green("Test passed")
