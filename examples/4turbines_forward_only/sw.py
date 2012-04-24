import configuration 
import numpy
from dirichlet_bc import DirichletBCSet
from reduced_functional import ReducedFunctional
from dolfin import *
set_log_level(PROGRESS)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.ConstantInflowPeriodicSidesPaperConfiguration()

# The turbine position is the control variable 
turbine_pos = [] 
border_x = 80.
border_y = 20.
for x_r in numpy.linspace(0.+border_x, config.params["basin_x"]-border_x, 2):
    for y_r in numpy.linspace(0.+border_y, config.params["basin_y"]-border_y, 2):
      turbine_pos.append((float(x_r), float(y_r)))

config.set_turbine_pos(turbine_pos)
info_blue("Deployed " + str(len(turbine_pos)) + " turbines.")

model = ReducedFunctional(config)
m0 = model.initial_control()
j0 = model.j(m0)
dj0 = model.dj(m0)
info("Power outcome: %f" % (j0, ))
info("Power gradient:" + str(dj0))
info("Timing summary:")
timer = Timer("NULL")
timer.stop()

list_timings()

