import configuration 
import numpy
from dirichlet_bc import DirichletBCSet
from reduced_functional import ReducedFunctional
from dolfin import *
set_log_level(PROGRESS)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.ConstantInflowPeriodicSidesPaperConfiguration(nx=100, ny=33)

# The turbine position is the control variable 
config.params["turbine_pos"] = [[100, 33]] 

info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
# Choosing a friction coefficient of > 0.25 ensures that overlapping turbines will lead to
# less power output.
config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))

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

