import configuration 
import numpy
import IPOptUtils
from reduced_functional import ReducedFunctional
from dolfin import *
from scipy.optimize import fmin_slsqp
set_log_level(INFO)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = configuration.ScenarioConfiguration("mesh.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params["controls"] = ["turbine_friction"]
config.params['automatic_scaling'] = False

# Place some turbines 
IPOptUtils.deploy_turbines(config, nx = 9, ny = 6)

model = ReducedFunctional(config, scaling_factor = -1, plot = True)
m0 = model.initial_control()
# Get the upper and lower bounds for the turbine friction
lb_f, ub_f = IPOptUtils.friction_constraints(config, lb = 0., ub = 2*config.turbine_friction)
bounds = [(lb_f[i], ub_f[i]) for i in range(len(lb_f))]

fmin_slsqp(model.j, m0, fprime = model.dj, bounds = bounds, iprint = 2, iter = 200)
