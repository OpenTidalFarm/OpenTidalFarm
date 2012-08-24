import sys
import configuration 
import shallow_water_model as sw_model
import finite_elements
from initial_conditions import SinusoidalInitialCondition
from dolfin import *
from dolfin_adjoint import *
import math 
from dirichlet_bc import DirichletBCSet
from domains import *

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
  sw_model.sw_solve(config, state, annotate=False, u_source = source)

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
  config.set_domain( RectangularDomain(640., 320., nx = 2*2**refinement_level, ny = 2*2**refinement_level), warning = False)

  # Choose more appropriate timing settings for the mms test
  config.params["k"] = pi/config.domain.basin_x
  config.params["start_time"] = 0
  config.params["finish_time"] = pi/(sqrt(config.params["g"]*config.params["depth"])*config.params["k"])/10
  config.params["dt"] = config.params["finish_time"]/300

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

  return error(config)

errors = []
tests = 5
for refinement_level in range(1, tests):
  errors.append(test(refinement_level))
# Compute the order of convergence 
conv = [] 
for i in range(len(errors)-1):
  conv.append(abs(math.log(errors[i+1]/errors[i], 2)))

# Plot the results
@cpu0only
def plot(hs, errors, file_name):
    scaling_factor = 2 * errors[-1]/(hs[-1]**2)
    second_order = [scaling_factor*h**2 for h in hs]
    pylab.figure()
    pylab.loglog(hs, second_order, 'g--', hs, errors, 'go-')
    pylab.legend(('Second order convergence', 'L2 norm of error'), 'lower right', shadow=True, fancybox=True)
    pylab.xlabel("Relative element size")
    pylab.ylabel("Model error")
    pylab.savefig(file_name)

hs = [1./2**h for h in range(len(errors))]

plot(hs, errors, "spatial_convergence.pdf")

info_green("Errors: %s.", str(errors))
info_green("Spatial order of convergence (expecting 2.0): %s.", str(conv))
if min(conv)<1.8:
  info_red("Spatial convergence test failed for wave_flather")
  sys.exit(1)
else:
  info_green("Test passed")
