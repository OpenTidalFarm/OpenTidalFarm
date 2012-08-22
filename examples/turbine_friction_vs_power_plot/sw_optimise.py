''' An example that plots the power ouput of a single turbine for different fricition values. (see also test_optimal_friction_single_turbine) '''
import sys
import configuration 
import numpy
import matplotlib.pyplot as plt
from dolfin import *
from dolfin_adjoint import *
from reduced_functional import ReducedFunctional
from helpers import info, info_green, info_red, info_blue
from scipy.optimize import fmin_slsqp, fmin_l_bfgs_b
set_log_level(ERROR)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
config = configuration.ScenarioConfiguration("mesh.xml", inflow_direction = [1, 0])
# Switch of the automatic scaling, since it currently does not support scaling of turbine friction. 
config.params['automatic_scaling'] = False

# Place one turbine 
basin_x = 640.
basin_y = 320.
turbine_pos = [[basin_x/3, basin_y/2]] 
config.set_turbine_pos(turbine_pos)
config.params['controls'] = ['turbine_friction']

# Use a negative scaling factor as we want to maximise the power output
model = ReducedFunctional(config, scaling_factor = -1)
m0 = model.initial_control()

#fmin_slsqp(model.j, m0, fprime = model.dj, iprint = 2, bounds = [(10., 50.)])
fmin_l_bfgs_b(model.j, m0, fprime = model.dj, iprint = 2)
