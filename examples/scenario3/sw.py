''' This example solves the forward model in an L shaped domain. '''
import configuration 
import numpy
from dirichlet_bc import DirichletBCSet
from reduced_functional import ReducedFunctional
from domains import GMeshDomain 
from dolfin import *
set_log_level(PROGRESS)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.ScenarioConfiguration("mesh.xml", inflow_direction = [0,1])

# Place some turbines 
basin_x = 1200
basin_y = 1000
land_x = 600
land_y = 300
land_site_delta = 100
site_x = 150
site_y = 100
site_x_start = basin_x - land_x
site_y_start = land_y + land_site_delta 

turbine_pos = []
for x_r in numpy.linspace(site_x_start + config.params["turbine_x"], site_x_start + site_x - config.params["turbine_y"], 3):
    for y_r in numpy.linspace(site_y_start + config.params["turbine_x"], site_y_start + site_y - config.params["turbine_y"], 3):
        turbine_pos.append((float(x_r), float(y_r)))
config.set_turbine_pos(turbine_pos)

info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")

model = ReducedFunctional(config, scaling_factor = -10**5)
m0 = model.initial_control()
j0 = model.j(m0)
dj0 = model.dj(m0)
print "Getting the adjoints a second time:"
j0 = model.j(m0)
dj0 = model.dj(m0)


