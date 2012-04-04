''' This test checks the correct implemetation of the turbine derivative terms.
    For that, we apply the Taylor remainder test on functional J(u, m) = <turbine_friction(m), turbine_friction(m)>,
    where m contains the turbine positions and the friction magnitude. 
'''

import sys
import configuration 
import numpy
from dolfin import *
from helpers import test_gradient_array
from reduced_functional import ReducedFunctional 

def default_config():
  numpy.random.seed(21) 
  config = configuration.DefaultConfiguration(nx=40, ny=20)
  config.params["dump_period"] = 1000
  config.params["verbose"] = 0

  # Turbine settings
  config.params["turbine_pos"] = [[1000., 500.], [1600, 300], [2500, 700]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 200
  config.params["turbine_y"] = 400

  return config

# run the taylor remainder test 
config = default_config()
model = ReducedFunctional(config)
m0 = model.initial_control()

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
p = numpy.random.rand(len(m0))

# Run with a functional that does not depend on m directly
for turbine_model, s in {'GaussianTurbine': {'seed': 100.0, 'tol': 1.9}, 'BumpTurbine': {'seed': 0.001, 'tol': 1.99}}.items():
  print "************* ", turbine_model, " ********************"
  config.params["turbine_model"] = turbine_model 
  minconv = test_gradient_array(model.j, model.dj, m0, s['seed'], perturbation_direction=p)

  if minconv < s['tol']:
    sys.exit(1)
