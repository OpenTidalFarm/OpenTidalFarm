import configuration 
import numpy
import IPOptUtils
from reduced_functional import ReducedFunctional
from dolfin import *
from scipy.optimize import fmin_slsqp
from helpers import info, info_green, info_red, info_blue
set_log_level(ERROR)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 

# Some domain information extracted from the geo file
basin_x = 1200
basin_y = 1000
land_x = 600
land_y = 300
land_site_delta = 100
site_x = 150
site_y = 100
site_x_start = basin_x - land_x
site_y_start = land_y + land_site_delta 
config = configuration.ScenarioConfiguration("mesh.xml", inflow_direction = [0,1])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines 
IPOptUtils.deploy_turbines(config, nx = 3, ny = 3) 

model = ReducedFunctional(config, scaling_factor = -10**1, plot = True)
m0 = model.initial_control()
j0 = model.j(m0)
dj0 = model.dj(m0)
info("Power outcome: %f" % (j0, ))
info("Power gradient:" + str(dj0))
info("Timing summary:")
timer = Timer("NULL")
timer.stop()
