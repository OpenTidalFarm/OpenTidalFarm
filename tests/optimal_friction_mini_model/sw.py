''' Test description:
 - single turbine (with constant friction distribution) whose size exceeds the size of the domain
 - constant velocity profile with an initial x-velocity of 2.
 - control: turbine friction
 - the mini model will compute a x-velocity of 2/(f + 1) where f is the turbine friction.
 - the functional is \int C * f * ||u||**3 where C is a constant
 - hence we maximise C * f * ( 2/(f + 1) )**3, f > 0 which has the solution f = 0.5

 Note: The solution is known only because we use a constant turbine friction distribution. 
       However this turbine model is not differentiable at its boundary, and this is why
       the turbine size has to exceed the domain.
 '''

import sys
from opentidalfarm import *
set_log_level(PROGRESS)

def default_config():
  config = configuration.DefaultConfiguration(nx=20, ny=10, finite_element=finite_elements.p1dgp2)
  config.params["verbose"] = 0

  # dt is used in the functional only, so we set it here to 1.0
  config.params["dt"] = 1.0
  # Turbine settings
  config.params["turbine_pos"] = [[500., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 1e10
  config.params["turbine_y"] = 1e10
  config.params['controls'] = ['turbine_friction']
  config.params["functional_final_time_only"] = True

  k = pi/config.domain.basin_x
  config.params['initial_condition'] = SinusoidalInitialCondition(config, 2.0, k, config.params['depth'])

  return config

config = default_config()
rf = ReducedFunctional(config, scale = 1e-3, forward_model=mini_model.mini_model_solve)
m0 = rf.initial_control()
rf(m0)
rf.dj(m0, forget=False)

p = numpy.random.rand(len(m0))
minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.001, perturbation_direction=p)
if minconv < 1.98:
  info_red("The gradient taylor remainder test failed.")
  sys.exit(1)

bounds = [0, 100]
maximize(rf, bounds=bounds, method="SLSQP", scale=1e-3) 

if abs(config.params["turbine_friction"][0]-0.5) > 10**-4: 
  info_red("The optimisation algorithm did not find the optimal friction value.")
  sys.exit(1) 
else:
  info_green("Test passed")
