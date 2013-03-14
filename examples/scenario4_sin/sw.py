from opentidalfarm import *
set_log_level(ERROR)

# Some domain information extracted from the geo file
basin_x = 1280.
basin_y = 640.+320.
site_x = 320.
site_y = 160.
rad = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y/2-rad)/2-site_y/2 
config = UnsteadyConfiguration("mesh.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)

config.info()

rf = ReducedFunctional(config, scaling_factor = -1, plot = True)
m0 = rf.initial_control()

lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config)
minimize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP")
