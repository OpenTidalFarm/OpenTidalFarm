import sys
from opentidalfarm import *
from dolfin_adjoint import adj_reset 
from math import log
from opentidalfarm.helpers import cpu0only
import opentidalfarm.domains
import pylab

set_log_level(ERROR)
parameters["std_out_all_processes"] = False;

def error(config,eta0,k):
  state = Function(config.function_space)
  state.interpolate(SinusoidalInitialCondition(config,eta0,k,config.params['depth']))

  adj_reset()
  shallow_water_model.sw_solve(config, state, annotate=False)

  analytic_sol = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", \
                             "0", \
                             "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"), \
                             eta0=eta0, g=config.params["g"], \
                             depth=config.params["depth"], t=config.params["current_time"], k=k)
  exactstate = Function(config.function_space)
  exactstate.interpolate(analytic_sol)
  e = state - exactstate
  return sqrt(assemble(dot(e,e)*dx))

def test(refinement_level):
    config = configuration.DefaultConfiguration(nx=2**refinement_level, ny=2*2**refinement_level) 
    config.set_domain(opentidalfarm.domains.RectangularDomain(3000, 1000, 2**refinement_level, 2*2**refinement_level))
    eta0 = 2.0
    k = pi/config.domain.basin_x
    config.params["finish_time"] = pi/(sqrt(config.params["g"]*config.params["depth"])*k)/20
    config.params["dt"] = config.params["finish_time"]/100
    config.params["dump_period"] = 100000
    config.params["bctype"] = "strong_dirichlet"
    bc = DirichletBCSet(config)

    expression = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), 
                            eta0=eta0, 
                            g=config.params["g"], 
                            depth=config.params["depth"], 
                            t=config.params["current_time"], 
                            k=k)

    bc.add_analytic_u(1, expression)
    bc.add_analytic_u(2, expression)
    bc.add_analytic_u(3, expression)
    config.params["strong_bc"] = bc

    return error(config, eta0, k)

errors = []
tests = 7
for refinement_level in range(2, tests):
  errors.append(test(refinement_level))
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

#plot(hs, errors, "spatial_convergence.pdf")

info_green("Absolute error values: %s" % str(errors))
info_green("Spatial order of convergence (expecting 2.0): %s" % str(conv))
if min(conv)<1.8:
  info_red("Spatial convergence test failed for wave_dirichlet")
  sys.exit(1)
else:
  info_green("Test passed")
