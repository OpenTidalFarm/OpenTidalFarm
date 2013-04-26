from opentidalfarm import *
import sys
import pickle
set_log_level(ERROR)

layer = int(sys.argv[1])
# Extract the settings that we use for this multigrid layer
#            Meshfile      Max iter     # Timesteps         
settings = (["mesh0.xml",        50,             50],
            ["mesh1.xml",        40,             20],
            ["mesh2.xml",        30,             25])[layer]

# Some domain information extracted from the geo file
basin_x = 1280.
basin_y = 640.+320.
site_x = 320.
site_y = 160.
rad = 160.
period = 12.*60*60

site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y/2-rad)/2-site_y/2 
eta0 = (2.0+1e-10)/sqrt(9.81/50) # This will give a inflow velocity of 2m/s
config = UnsteadyConfiguration(settings[0], inflow_direction = [1, 0], eta0=eta0, period=period)

# Update the timestep according to the layer-depending setting
config.params['dt'] = period/settings[2] 


config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params['initial_condition'] = ConstantFlowInitialCondition(config) 
config.params['save_checkpoints'] = True

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)
# If available, overload the turbine positions from the coarser layer 
try:
    config.params["turbine_pos"] = pickle.load(open("turbine_pos"+str(layer+1)+".dat", "r"))
except IOError:
    print "No turbine positions from coarser level found. Starting from the initial layout."

config.info()

rf = ReducedFunctional(config)

# Load checkpoints if desired by the user
try:
	rf.load_checkpoint("checkpoint")
except IOError:
	print "No checkpoints file found."

lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config)
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP", options = {"maxiter": settings[1]})

# Save most recent turbine positions to file
pickle.dump(config.params["turbine_pos"], open("turbine_pos"+str(layer)+".dat", "w"))
