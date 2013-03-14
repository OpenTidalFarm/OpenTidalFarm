from opentidalfarm import *
set_log_level(ERROR)

config = SteadyConfiguration("mesh/earth_orkney_converted_coarse.xml", inflow_direction=[0.9865837220518425, -0.16325611591095968]) 
config.params['diffusion_coef'] = 90.0
config.params['turbine_x'] = 40.
config.params['turbine_y'] = 40.

# Some domain information extracted from the geo file
site_x = 1000.
site_y = 500.
site_x_start = 1.03068e+07 
site_y_start = 6.52246e+06 - site_y 

config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)
config.params["turbine_friction"] = 0.5*numpy.array(config.params["turbine_friction"]) 

rf = ReducedFunctional(config, plot = True)
m0 = rf.initial_control()

# Get the upper and lower bounds for the turbine positions
lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config)
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP") 
