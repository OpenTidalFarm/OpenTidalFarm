from opentidalfarm import *
set_log_level(INFO)

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = SteadyConfiguration("mesh.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params["controls"] = ["turbine_friction"]
config.params['automatic_scaling'] = False

deploy_turbines(config, nx = 8, ny = 6)
rf = ReducedFunctional(config)

config.info()

m0 = rf.initial_control()
lb_f, ub_f = friction_constraints(config, lb = 0., ub = 21.)
maximize(rf, bounds = [lb_f, ub_f], method = 'SLSQP', options = {'maxiter': 200})
