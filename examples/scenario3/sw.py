from opentidalfarm import *
set_log_level(ERROR)

# Some domain information extracted from the geo file
basin_x = 1600
land_x = 640
land_y = 320
land_site_delta = 100
site_x = 320
site_y = 160

site_x_start = basin_x - land_x
site_y_start = land_y + land_site_delta 
config = SteadyConfiguration("mesh.xml", inflow_direction = [0, 1])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)

config.info()

rf = ReducedFunctional(config)

lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config)
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP", options = {"maxiter": 200})
