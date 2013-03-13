''' Test description:
 - single turbine (with constant friction distribution) whose size exceeds the size of the domain
 - constant velocity profile with an initial x-velocity of 2.
 - control: turbine friction
 - the mini model will compute a x-velocity of 2/(f + 1) wher ef is the turbine friction.
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
  config = configuration.DefaultConfiguration(nx=20, ny=10, finite_element = finite_elements.p1dgp2)
  config.params["verbose"] = 0

  # dt is used in the functional only, so we set it here to 1.0
  config.params["dt"] = 1.0
  # Turbine settings
  config.params["turbine_pos"] = [[500., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 8000
  config.params["turbine_y"] = 8000

  return config

config = default_config()
config.params["save_checkpoints"] = True

rf = ReducedFunctional(config, scaling_factor = -10**-3, forward_model = mini_model.mini_model_solve)
m0 = rf.initial_control()

if len(sys.argv) > 1 and sys.argv[1] == "--from-checkpoint":
  rf.load_checkpoint("checkpoint")

# If this option does not produce any ipopt outputs, delete the ipopt.opt file
g = lambda m: []
dg = lambda m: []

lb_f, ub_f = friction_constraints(config, lb = 0., ub = 100.)
bb = [Constant(500)]*2
bounds = [lb_f + bb, ub_f + bb]
m = minimize(rf, bounds = bounds, method = "SLSQP", options={'maxiter': 5}) 
