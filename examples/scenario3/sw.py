import numpy
from opentidalfarm import *
set_log_level(ERROR)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 

# Some domain information extracted from the geo file
basin_x = 1600
land_x = 640
land_y = 320
land_site_delta = 100
site_x = 320
site_y = 160

site_x_start = basin_x - land_x
site_y_start = land_y + land_site_delta 
config = ScenarioConfiguration("mesh.xml", inflow_direction = [0,1])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)

config.info()

rf = ReducedFunctional(config, scaling_factor = -1, plot = True)
m0 = rf.initial_control()

lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config)
minimize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP")
