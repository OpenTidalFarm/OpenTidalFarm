''' This tests checks that the power output of a single turbine in a periodic domain is independent of its position ''' 
import sys
import configuration 
import numpy
from dirichlet_bc import DirichletBCSet
import IPOptUtils
import ipopt
from helpers import test_gradient_array
from reduced_functional import ReducedFunctional
from dolfin import *
set_log_level(ERROR)

if len(sys.argv) not in [2, 3]:
    print "Missing command line argument: y position of the turbine"
    print "Usage: sw.py [--fine] y_position"
    sys.exit(1)

turbine_y_pos = float(sys.argv[-1])

nx = 100
ny = 33
basin_x = 2*nx
basin_y = 2*ny
if len(sys.argv) == 3 and sys.argv[1] == '--fine':
    nx *= 2
    ny *= 2

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = configuration.PaperConfiguration(nx = nx, ny = ny, basin_x = basin_x, basin_y = basin_y) 
  config.params["dump_period"] = 1
  config.params["verbose"] = 0

  config.params["finish_time"] = 3.*period/4 

  bc = DirichletBCSet(config)
  bc.add_analytic_u(config.left)
  bc.add_analytic_u(config.right)
  bc.add_periodic_sides()
  config.params["strong_bc"] = bc

  # The turbine position is the control variable 
  config.params["turbine_pos"] = [[config.params["basin_x"]/2, turbine_y_pos]] 
  info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")

  # Choosing a friction coefficient of > 0.02 ensures that overlapping turbines will lead to
  # less power output.
  config.params["turbine_friction"] = 0.2*numpy.ones(len(config.params["turbine_pos"]))

  return config

config = default_config()
model = ReducedFunctional(config, scaling_factor = 10**-4, plot = True)
m0 = model.initial_control()
print "Functional value for m0 = ", m0, ": ", model.j(m0)
print "Derivative value for m0 = ", m0, ": ", model.dj(m0)
