''' This test checks the correctness of the gradient of the turbine position with respect to its position '''
import sys
from opentidalfarm import *
import opentidalfarm.domains
set_log_level(INFO)

def default_config():
  config = configuration.DefaultConfiguration(nx=30, ny=10)
  config.set_domain(opentidalfarm.domains.RectangularDomain(3000, 1000, 30, 10))
  period = 1.24*60*60 # Wave period
  info("Wave period (in h): %f" % (period/60/60) )
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4
  config.params["dt"] = period/50
  config.params["finish_time"] = config.params["start_time"] + 2*config.params["dt"]
  config.params["theta"] = 0.6
  config.params["include_advection"] = True 
  config.params["include_viscosity"] = True 
  config.params["viscosity"] = 20.0
  config.params["controls"] = ["turbine_pos"]

  config.info()
  # Boundary condition settings
  config.params["bctype"] = "strong_dirichlet"

  k = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  eta0 = 2
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
  config.params["strong_bc"] = bc

  # Initial condition
  config.params["initial_condition"] = SinusoidalInitialCondition(config, eta0, k, config.params["depth"])


  # Turbine settings
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
rf = ReducedFunctional(config)
m0 = rf.initial_control()

p = numpy.random.rand(len(m0))
minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.1, perturbation_direction=p)
if minconv < 1.98:
    info_red("The gradient taylor remainder test failed.")
    sys.exit(1)
else:
    info_green("Test passed")
    
