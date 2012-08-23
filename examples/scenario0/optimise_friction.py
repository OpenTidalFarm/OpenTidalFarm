''' Plots the power ouput of a single turbine for different fricition values. (see also test_optimal_friction_single_turbine) '''
import configuration 
import numpy
import matplotlib.pyplot as plt
from dolfin import *
from dolfin_adjoint import *
from reduced_functional import ReducedFunctional
from helpers import info, info_green, info_red, info_blue
from scipy.optimize import fmin_slsqp, fmin_l_bfgs_b
set_log_level(ERROR)

basin_x = 640.
basin_y = 320.

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
config = configuration.ScenarioConfiguration("mesh.xml", inflow_direction = [1, 0])
config.params['automatic_scaling'] = False

# Place one turbine 
offset = 0.0
turbine_pos = [[basin_x/3 + offset, basin_y/2 + offset]] 
info_green("Turbine position: " + str(turbine_pos))
config.set_turbine_pos(turbine_pos)
config.params['controls'] = ['turbine_friction']

# Use a negative scaling factor as we want to maximise the power output
model = ReducedFunctional(config, scaling_factor = -1)
m0 = model.initial_control()

fmin_l_bfgs_b(model.j, m0, fprime = model.dj, iprint = 2)
