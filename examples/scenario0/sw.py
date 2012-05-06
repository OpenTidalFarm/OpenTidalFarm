import configuration 
import numpy
from reduced_functional import ReducedFunctional
from dolfin import *
set_log_level(ERROR)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
config = configuration.ScenarioConfiguration("mesh.xml", inflow_direction = [1, 0])
# Switch of the automatic scaling, since we will not solve an optimisation problem
config.params['automatic_scaling'] = False

# Place some turbines 
basin_x = 640
basin_y = 320
turbine_pos = [[basin_x/3, basin_y/2]] 
config.set_turbine_pos(turbine_pos)


model = ReducedFunctional(config)
m0 = model.initial_control()
j0 = model.j(m0)
info("Power outcome: %f" % (j0, ))
