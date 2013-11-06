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
config.params["automatic_scaling"] = False
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)

config.info()

V = VectorFunctionSpace(Mesh("mesh.xml"), "CG", 2)
flow = project(Expression(("2.0", "0.0")), V)
wake_model = AnalyticalWakeModel(config, flow, model_type="Simple")
rf = ReducedFunctional(config, forward_model=wake_model)

lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config)
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP", options = {"maxiter": 200})
