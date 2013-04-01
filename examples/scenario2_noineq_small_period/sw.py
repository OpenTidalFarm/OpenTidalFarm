from opentidalfarm import *
set_log_level(PROGRESS)

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
eta0 = (2.0+1e-10)/sqrt(9.81/50) # This will give a inflow velocity of 2m/s
config = UnsteadyConfiguration("mesh.xml", inflow_direction = [1,0], period = 10.*60, eta0=eta0)
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params['initial_condition'] = ConstantFlowInitialCondition(config)


# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)

config.info()

rf = ReducedFunctional(config)
bounds = position_constraints(config) 

maximize(rf, bounds=bounds, method = "SLSQP", options = {"maxiter": 400})
