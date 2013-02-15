import configuration 
import numpy
import IPOptUtils
from reduced_functional import ReducedFunctional
from dolfin import *
from scipy.optimize import fmin_slsqp
from helpers import info, info_green, info_red, info_blue
set_log_level(PROGRESS)
from dirichlet_bc import DirichletBCSet

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = configuration.SinusoidalScenarioConfiguration("mesh.xml", inflow_direction = [1,0], period = 10.*60)
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines 
IPOptUtils.deploy_turbines(config, nx = 8, ny = 4)

model = ReducedFunctional(config, scaling_factor = -1, plot = True)
m0 = model.initial_control()

# Get the upper and lower bounds for the turbine positions
lb, ub = IPOptUtils.position_constraints(config) 
bounds = [(float(lb[i]), float(ub[i])) for i in range(len(lb))]

fmin_slsqp(model.j, m0, fprime = model.dj, bounds = bounds, iprint = 2, iter = 200)
