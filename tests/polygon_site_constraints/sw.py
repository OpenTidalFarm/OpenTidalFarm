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
import opentidalfarm.domains

def default_config():
  config = configuration.DefaultConfiguration(nx=40, ny=20, finite_element = finite_elements.p1dgp2)
  config.set_domain(opentidalfarm.domains.RectangularDomain(3000, 1000, 40, 20))
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
rf = ReducedFunctional(config, forward_model = mini_model_solve)
m0 = rf.initial_control()

config.info()

ineq = generate_site_constraints(config, [[20, 20], [120, 20], [120, 220], [20, 220]])
maximize(rf, constraints=ineq, method="SLSQP", options={"maxiter": 2}) 

m = config.params["turbine_pos"][0]

info("Solution of the primal variables: m=" + repr(m) + "\n")

exit_code = 1
tol = 1
if abs(m[0]-120) > tol:
    info_red("The optimisation algorithm did not find the optimal x position: %f instead of 1500." % m[0])
elif abs(m[1]-220) > tol:
    info_red("The optimisation algorithm did not find the optimal y position: %f instead of 500." %m[1])
else:
    info_green("Test passed")
    exit_code = 0

sys.exit(exit_code) 
