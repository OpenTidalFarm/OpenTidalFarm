''' Test description
 - single turbine
 - bubble velocity profile with maximum in the center of the domain
 - control: turbine position
 - the optimal placement for the turbine is where the velocity profile reaches its maximum (the center of the domain)
'''

import sys
from opentidalfarm import *
from opentidalfarm.helpers import test_gradient_array
from opentidalfarm.mini_model import mini_model_solve

def default_config():
  config = configuration.DefaultConfiguration(nx=40, ny=20, finite_element = finite_elements.p1dgp2)
  config.params["verbose"] = 0

  # dt is used in the functional only, so we set it here to 0.8
  config.params["dt"] = 0.8
  # Turbine settings
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [[500., 200.]]
  config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 800
  config.params["turbine_y"] = 800
  config.params["controls"] = ['turbine_pos']
  config.params["initial_condition"] = BumpInitialCondition(config)
  config.params["automatic_scaling"] = True
  
  return config

config = default_config()
rf = ReducedFunctional(config, forward_model=mini_model_solve)
m0 = rf.initial_control()

config.info()

p = numpy.random.rand(len(m0))
minconv = test_gradient_array(rf.j, rf.dj, m0, seed=0.005, perturbation_direction=p)
if minconv < 1.9:
  info_red("The gradient taylor remainder test failed.")
  sys.exit(1)

# If this option does not produce any ipopt outputs, delete the ipopt.opt file
g = lambda m: []
dg = lambda m: []

bounds = [[Constant(0), Constant(0)], [Constant(3000), Constant(1000)]] 

maximize(rf, bounds = bounds, method = "SLSQP") 

m = config.params["turbine_pos"][0]

info("Solution of the primal variables: m=" + repr(m) + "\n")

print m

exit_code = 1
if abs(m[0]-1500) > 40:
    info_red("The optimisation algorithm did not find the optimal x position: %f instead of 1500." % m[0])
elif abs(m[1]-500) > 0.4:
    info_red("The optimisation algorithm did not find the optimal y position: %f instead of 500." %m[1])
else:
    info_green("Test passed")
    exit_code = 0

sys.exit(exit_code) 
