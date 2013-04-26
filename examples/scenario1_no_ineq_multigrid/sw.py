from opentidalfarm import *
import sys
import pickle
set_log_level(INFO)

layer = int(sys.argv[1])
# Extract the settings that we use for this multigrid layer
#            Meshfile      Max iter           
settings = (["mesh0.xml",        50],
            ["mesh1.xml",        40],
            ["mesh2.xml",        30])[layer]

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = SteadyConfiguration(settings[0], inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
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
maximize(rf, bounds = [lb, ub], method = "SLSQP", options = {"maxiter": settings[1]})

# Save most recent turbine positions to file
pickle.dump(config.params["turbine_pos"], open("turbine_pos"+str(layer)+".dat", "w"))
