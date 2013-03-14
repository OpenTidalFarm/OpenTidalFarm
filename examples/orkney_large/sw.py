from opentidalfarm import *
set_log_level(PROGRESS)

# Some domain information extracted from the geo file
site_x = 2000.
site_y = 1000.
site_x_start = 1.03068e+07 
site_y_start = 6.52276e+06 - site_y 

inflow_x = 8400.
inflow_y = -1390.
inflow_norm = (inflow_x**2 + inflow_y**2)**0.5
inflow_direction = [inflow_x/inflow_norm, inflow_y/inflow_norm]
print "inflow_direction: ", inflow_direction

config = SteadyConfiguration("mesh/earth_orkney_converted.xml", inflow_direction = inflow_direction) 
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params['diffusion_coef'] = 90.0
config.params['save_checkpoints'] = True
#config.params['linear_solver'] = 'superlu_dist' 

# Place some turbines 
deploy_turbines(config, nx = 16, ny = 8)
config.params["turbine_friction"] = 0.5*numpy.array(config.params["turbine_friction"]) 

rf = ReducedFunctional(config, scaling_factor = -1, plot = True)
m0 = rf.initial_control()

# Load checkpoints if desired by the user
if len(sys.argv) > 1 and sys.argv[1] == "--from-checkpoint":
  rf.load_checkpoint("checkpoint")

# Get the upper and lower bounds for the turbine positions
lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config)
minimize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP") 
