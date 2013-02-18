import configuration 
import numpy
import IPOptUtils
from reduced_functional import ReducedFunctional
from dolfin import *
from dolfin_adjoint import minimize
set_log_level(PROGRESS)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 

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

config = configuration.ScenarioConfiguration("mesh/earth_orkney_converted.xml", inflow_direction = inflow_direction) 
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params['diffusion_coef'] = 90.0
#config.params['linear_solver'] = 'superlu_dist' 

# Place some turbines 
IPOptUtils.deploy_turbines(config, nx = 16, ny = 8)
config.params["turbine_friction"] = 0.5*numpy.array(config.params["turbine_friction"]) 

rf = ReducedFunctional(config, scaling_factor = -1, plot = True)
m0 = rf.initial_control()

# Get the upper and lower bounds for the turbine positions
lb, ub = IPOptUtils.position_constraints(config) 
ineq = IPOptUtils.get_minimum_distance_constraint_func(config)
minimize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP") 
