import sys
import configuration 
import shallow_water_model as sw_model
import finite_elements
from dirichlet_bc import DirichletBCSet
from initial_conditions import SinusoidalInitialCondition
from dolfin import *
from math import log
from helpers import cpu0only
import pylab

set_log_level(ERROR)
parameters["std_out_all_processes"] = False;

def error(config):
  state = Function(config.function_space)
  state.interpolate(SinusoidalInitialCondition(config)())

  sw_model.sw_solve(config, state, annotate=False)

  analytic_sol = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", \
                             "0", \
                             "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"), \
                             eta0=config.params["eta0"], g=config.params["g"], \
                             depth=config.params["depth"], t=config.params["current_time"], k=config.params["k"])
  exactstate = Function(config.function_space)
  exactstate.interpolate(analytic_sol)
  e = state - exactstate
  return sqrt(assemble(dot(e,e)*dx))

def test(refinment_level):
    config = configuration.DefaultConfiguration(nx=2**refinment_level, ny=2*2**refinment_level) 
    config.params["finish_time"] = pi/(sqrt(config.params["g"]*config.params["depth"])*config.params["k"])/20
    config.params["dt"] = config.params["finish_time"]/100
    config.params["dump_period"] = 100000
    config.params["bctype"] = "strong_dirichlet"
    bc = DirichletBCSet(config)
    bc.add_analytic_u(config.domain.left)
    bc.add_analytic_u(config.domain.right)
    bc.add_analytic_u(config.domain.sides)
    config.params["strong_bc"] = bc

    return error(config)

errors = []
tests = 7
for refinment_level in range(2, tests):
  errors.append(test(refinment_level))
# Compute the order of convergence 
conv = [] 
for i in range(len(errors)-1):
  conv.append(abs(log(errors[i+1]/errors[i], 2)))

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

info_green("Absolute error values: %s" % str(errors))
info_green("Spatial order of convergence (expecting 2.0): %s" % str(conv))
if min(conv)<1.8:
  info_red("Spatial convergence test failed for wave_dirichlet")
  sys.exit(1)
else:
  info_green("Test passed")
