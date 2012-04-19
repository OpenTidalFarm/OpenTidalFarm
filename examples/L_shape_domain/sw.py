''' This example solves the forward model in an L shaped domain. '''
import configuration 
import numpy
from dirichlet_bc import DirichletBCSet
from reduced_functional import ReducedFunctional
from domains import LShapeDomain
from dolfin import *
set_log_level(PROGRESS)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.ConstantInflowPeriodicSidesPaperConfiguration()

L_len = 50
config.set_domain( LShapeDomain("mesh.xml", L_len) )
# We also need to reapply the bc
bc = DirichletBCSet(config)
bc.add_constant_flow(config.domain.left)
bc.add_noslip_u(config.domain.sides)
config.params['strong_bc'] = bc

# Place some 
config.params["turbine_pos"] = [[L_len/4, L_len/4]] 

info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
# Choosing a friction coefficient of > 0.25 ensures that overlapping turbines will lead to
# less power output.
config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))

model = ReducedFunctional(config, scaling_factor = -10**-6)
m0 = model.initial_control()
j0 = model.j(m0)
dj0 = model.dj(m0)
info("Power outcome: %f" % (j0, ))
info("Power gradient:" + str(dj0))
info("Timing summary:")
timer = Timer("NULL")
timer.stop()

list_timings()

