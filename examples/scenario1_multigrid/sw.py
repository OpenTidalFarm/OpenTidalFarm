from opentidalfarm import *
from optparse import OptionParser
import sys
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

config.info()

rf = ReducedFunctional(config)

# Load checkpoints if desired by the user
try:
	rf.load_checkpoint("checkpoint")
except IOError:
	print "No checkpoints file found."

lb, ub = position_constraints(config) 
maximize(rf, bounds = [lb, ub], method = "SLSQP", options = {"maxiter": settings[1]})
