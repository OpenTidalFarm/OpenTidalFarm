''' This example optimises the position of three turbines 
    using the shallow water solver. '''

import sys
from opentidalfarm import *
from dolfin import log, INFO, ERROR

config = UnsteadyConfiguration("mesh.xml", inflow_direction = [1, 1])
config.params['finish_time'] = config.params["start_time"] + 2*config.params["dt"]

# Deploy some turbines 
turbine_pos = [] 
# The configuration does not converge for this (admittely unphysical) setup, so we help a little with some viscosity
#config.params['viscosity'] = 40.0
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = basin_x - site_x/2
site_y_start = basin_y - site_y/2
config.params['turbine_x'] = 50. 
config.params['turbine_y'] = 50. 
config.params['controls'] = ["dynamic_turbine_friction"]
config.params["automatic_scaling"] = False
#config.params['k'] = pi / basin_x

for x_r in numpy.linspace(site_x_start, site_x_start + site_x, 2):
    for y_r in numpy.linspace(site_y_start, site_y_start + site_y, 2):
      turbine_pos.append((float(x_r), float(y_r)))

config.set_turbine_pos(turbine_pos, friction=1.0)
info_blue("Deployed " + str(len(turbine_pos)) + " turbines.")

config.params["turbine_friction"] = [config.params["turbine_friction"]]*3

rf = ReducedFunctional(config, scale = 10**-6)
m0 = rf.initial_control()

rf.j(m0)

p = numpy.random.rand(len(m0))
seed = 0.1
#minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=seed, perturbation_direction=p, plot_file="convergence.pdf")
minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=seed, perturbation_direction=p)
if minconv < 1.9:
    log(ERROR, "The gradient taylor remainder test failed.")
    sys.exit(1)
else:
    log(INFO, "The gradient taylor remainder test passed.")
