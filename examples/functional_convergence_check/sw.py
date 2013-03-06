''' This tests checks that the power output of a single turbine in a periodic domain is independent of its position ''' 
import sys
from opentidalfarm import *
import numpy
set_log_level(ERROR)

if len(sys.argv) not in [3, 4]:
    print "Missing command line argument: y position of the turbine"
    print "Usage: sw.py [--fine] x_position y_position"
    sys.exit(1)

turbine_x_pos = float(sys.argv[-2])
turbine_y_pos = float(sys.argv[-1])
print "turbine_x_pos: ", turbine_x_pos
print "turbine_y_pos: ", turbine_y_pos

nx = 100
ny = 33
basin_x = 2*nx
basin_y = 2*ny
if len(sys.argv) == 4 and sys.argv[1] == '--fine':
    nx *= 2
    ny *= 2

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
# Force the basin size to 200x66 independent of nx and ny
config = ConstantInflowPeriodicSidesPaperConfiguration(nx = nx, ny = ny, basin_x = basin_x, basin_y = basin_y) 

turbine_pos = [[turbine_x_pos, turbine_y_pos]] 
config.set_turbine_pos(turbine_pos)
info_blue("Deployed " + str(len(turbine_pos)) + " turbines at positions " + str(turbine_pos) + " .")

model = ReducedFunctional(config, scaling_factor = 10**-6)
m0 = model.initial_control()
print "Functional value for m0 = ", m0, ": ", model.j(m0)
