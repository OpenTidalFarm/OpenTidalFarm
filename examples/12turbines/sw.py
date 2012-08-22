''' This example optimises the position of three turbines using the hallow water model. '''
import configuration 
import numpy
import IPOptUtils
from animated_plot import *
from reduced_functional import ReducedFunctional
from dolfin import *
from scipy.optimize import fmin_slsqp
set_log_level(ERROR)

basin_x = 100.
basin_y = 50.

# An animated plot to visualise the development of the functional value
plot = AnimatedPlot(xlabel='Iteration', ylabel='Functional value')

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 

# Some domain information extracted from the geo file
basin_x = 100.
basin_y = 50.
site_x = 100.
site_y = 50.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = configuration.ScenarioConfiguration("mesh.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines 
IPOptUtils.deploy_turbines(config, nx = 3, ny = 2)

model = ReducedFunctional(config, scaling_factor = -1, plot = True)
m0 = model.initial_control()

# Get the upper and lower bounds for the turbine positions
lb, ub = IPOptUtils.position_constraints(config) 
bounds = [(lb[i], ub[i]) for i in range(len(lb))]
#fmin_l_bfgs_b(model.j, m0, fprime = model.dj, bounds = bounds)
fmin_slsqp(model.j, m0, fprime = model.dj, bounds = bounds, iprint = 2)
