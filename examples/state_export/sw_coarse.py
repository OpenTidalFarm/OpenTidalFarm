''' This example demonstrates the use of a state writer callback function,
which allows to perform diagnostics on the shallow water solutions at each 
timestep and optimisation iteration. '''
from opentidalfarm import *
set_log_level(INFO)

def state_writer(state, u_p1, p_p1, it, optit):
    ''' This function is called after each timestep in the shallow water model
    and can be used for example to output the solution into a user-specific 
    format.
    The function parameters are:
        u: The velocity solution
        p: the free-surface elevation
        it: the timestep
        optit: the optimisation iteration
    '''
    print "Saving velocity/pressure solution as xyz data..."
    File("xyz_data/p_opiter_%i_timstep_%i_.xyz" % (optit, it)) << p_p1
    # Split u into the x and y component since xyz can only store scalar fields
    ux, uy = u_p1.split(deepcopy=True)
    File("xyz_data/ux_opiter_%i_timstep_%i_.xyz" % (optit, it)) << ux
    File("xyz_data/uy_opiter_%i_timstep_%i_.xyz" % (optit, it)) << uy

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = SteadyConfiguration("mesh_coarse.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Activate the custom state writer object
config.statewriter_callback = state_writer

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)

config.info()

rf = ReducedFunctional(config)

lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config)
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP", options={'maxiter':3})
