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
config.params["controls"] = ["turbine_friction", "turbine_pos"]

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)

config.info()

rf = ReducedFunctional(config)
# Get the upper and lower bounds for the turbine positions and friction
lb_f, ub_f = friction_constraints(config, lb = 0., ub = 21.)
lb, ub = position_constraints(config) 
# The first part of the control vector consists of the turbine friction values followed by their positions
bounds = [lb_f + lb, ub_f + ub]

ineq = get_minimum_distance_constraint_func(config)

maximize(rf, bounds = bounds, constraints = ineq, method = "SLSQP", options = {"maxiter": 300}) 
