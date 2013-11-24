import sys
from opentidalfarm import *
from dolfin_adjoint import adj_reset
from convergence_plot import *
import math 

set_log_level(ERROR)
parameters["std_out_all_processes"] = False;

def error(config, eta0, k):
  state = Function(config.function_space)
  state.interpolate(SinusoidalInitialCondition(config, eta0, k, config.params["depth"]))
  u_exact = "eta0*sqrt(g/depth) * cos(k*x[0]-sqrt(g*depth)*k*t)" 
  du_exact = "(- eta0*sqrt(g/depth) * sin(k*x[0]-sqrt(g*depth)*k*t) * k)"
  ddu_exact = "(diffusion_coef * eta0*sqrt(g/depth) * cos(k*x[0]-sqrt(g*depth)*k*t) * k*k)"
  eta_exact = "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"
  friction = "friction/depth * " + u_exact + "*pow(pow(" + u_exact + ", 2), 0.5)"
  # The source term
  source = Expression((u_exact + " * " + du_exact + " + " + ddu_exact + " + " + friction, 
                             "0.0"), \
                             eta0 = eta0, g = config.params["g"], \
                             depth = config.params["depth"], t = config.params["current_time"], \
                             k = k, diffusion_coef = config.params["diffusion_coef"],
                             friction = config.params["friction"])

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
  enorm = sqrt(assemble(dot(e,e)*dx))
  print enorm
  return enorm 

def test(refinement_level):
  # Set up the configuration
  config = configuration.UnsteadyConfiguration("mesh_dummy.xml", [1, 0]) 
  config.set_domain( RectangularDomain(640., 320., nx = 2*2**refinement_level, ny = 2*2**refinement_level), warning = False)

  # Choose more appropriate timing settings for the mms test
  k = pi/config.domain.basin_x
  config.params["start_time"] = 0.0
  config.params["finish_time"] = pi/(sqrt(config.params["g"]*config.params["depth"])*k)/10.0
  config.params["dt"] = config.params["finish_time"]/300.0

  # Make sure that we apply the analytical boundary conditions
  eta0 = 2.0
  config.params['initial_condition'] = SinusoidalInitialCondition(config, eta0, k, config.params['depth'])
  config.params["dump_period"] = 100000
  config.params["bctype"] = "strong_dirichlet"
  expression = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), 
                          eta0=eta0, 
                          g=config.params["g"], 
                          depth=config.params["depth"], 
                          t=config.params["current_time"], 
                          k=k)

  bc = DirichletBCSet(config)
  bc.add_analytic_u(1, expression)
  bc.add_analytic_u(2, expression)
  bc.add_analytic_u(3, expression)
  config.params['strong_bc'] = bc

  return config.domain.mesh.hmax(), error(config, eta0, k)

errors = []
element_sizes = []
tests = 5
for refinement_level in range(1, tests):
  element_size, e = test(refinement_level)
  errors.append(e)
  element_sizes.append(element_size)
# Compute the order of convergence 
conv = [] 
for i in range(len(errors)-1):
  conv.append(abs(math.log(errors[i+1]/errors[i], 2)))

# Plot the results
#save_convergence_plot(errors, element_sizes, "Spatial rate of convergence", "Spatial error", order = 2.0, show_title = False, xlabel = "Element size [m]")

info_green("Errors: %s.", str(errors))
info_green("Spatial order of convergence (expecting 2.0): %s.", str(conv))
if min(conv)<1.8:
  info_red("Spatial convergence test failed for wave_flather")
  sys.exit(1)
else:
  info_green("Test passed")
