import configuration 
import numpy
import IPOptUtils
from reduced_functional import ReducedFunctional
from dolfin import *
from dolfin_adjoint import minimize
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
config = configuration.SinusoidalScenarioConfiguration("mesh.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines 
IPOptUtils.deploy_turbines(config, nx = 8, ny = 4)

config.info()

rf = ReducedFunctional(config, scaling_factor = -1, plot = True)
m0 = rf.initial_control()

lb, ub = IPOptUtils.position_constraints(config) 
ineq = IPOptUtils.get_minimum_distance_constraint_func(config)
minimize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP", options = {"maxiter": 200})
