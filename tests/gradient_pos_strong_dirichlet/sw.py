''' This test checks the correctness of the gradient of the turbine position with respect to its position '''
import sys
from opentidalfarm import *
set_log_level(PROGRESS)

def default_config():
  config = configuration.DefaultConfiguration(nx=30, ny=10)
  period = 1.24*60*60 # Wave period
  k = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  info("Wave period (in h): %f" % (period/60/60) )
  config.params["dump_period"] = 10000
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4
  config.params["dt"] = period/50
  config.params["finish_time"] = config.params["start_time"] + 2*config.params["dt"]
  config.params["theta"] = 0.6
  config.params["include_advection"] = True 
  config.params["include_diffusion"] = True 
  config.params["diffusion_coef"] = 20.0
  config.params["newton_solver"] = True 
  config.params["picard_iterations"] = 20
  config.params["linear_solver"] = "default"
  config.params["preconditioner"] = "default"
  config.params["controls"] = ["turbine_pos"]

  # Boundary condition settings
  config.params["bctype"] = "strong_dirichlet"
  expression = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), eta0 = config.params["eta0"], g = config.params["g"], depth = config.params["depth"], t = config.params["current_time"], k = config.params["k"])
  bc = DirichletBCSet(config)
  bc.add_analytic_u(1, expression)
  bc.add_analytic_u(2, expression)
  bc.add_analytic_u(3, expression)
  config.params["strong_bc"] = bc

  # Turbine settings
  config.params["quadratic_friction"] = True
  config.params["friction"] = 0.0025
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [[1000, 400], [2000, 600]] 
  # Choosing a friction coefficient of > 0.02 ensures that overlapping turbines will lead to
  # less power output.
  config.params["turbine_friction"] = 0.2*numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 400
  config.params["turbine_y"] = 400

  return config

config = default_config()
model = ReducedFunctional(config)
m0 = model.initial_control()

p = numpy.random.rand(len(m0))
minconv = helpers.test_gradient_array(model.j, model.dj, m0, seed=0.1, perturbation_direction=p)
if minconv < 1.98:
    info_red("The gradient taylor remainder test failed.")
    sys.exit(1)
else:
    info_green("Test passed")
    
