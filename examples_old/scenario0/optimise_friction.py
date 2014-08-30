''' Plots the power ouput of a single turbine for different fricition values. (see also test_optimal_friction_single_turbine) '''
from opentidalfarm import *
set_log_level(ERROR)

basin_x = 640.
basin_y = 320.

config = SteadyConfiguration("mesh.xml", inflow_direction = [1, 0])
config.params['automatic_scaling'] = False

# Place one turbine 
offset = 0.0
turbine_pos = [[basin_x/3 + offset, basin_y/2 + offset]] 
info_green("Turbine position: " + str(turbine_pos))
config.set_turbine_pos(turbine_pos)
config.params['controls'] = ['turbine_friction']

# Use a negative scaling factor as we want to maximise the power output
rf = ReducedFunctional(config)
maximize(rf)

print "The optimal turbine friction is ", config.params["turbine_friction"][0]
