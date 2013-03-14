''' This example optimises the position of three turbines using the hallow water model. '''

import sys
from opentidalfarm import *
set_log_level(ERROR)

for c in [DefaultConfiguration, SteadyConfiguration]:
    info_green("Testing configuration " + c.__name__)
    if c == SteadyConfiguration:
        config = c("mesh.xml", inflow_direction = [1, 1])
    else:
        config = c(nx = 15, ny = 15)
    config.params['finish_time'] = config.params["start_time"] + 2*config.params["dt"]

    # Deploy some turbines 
    turbine_pos = [] 
    if c == SteadyConfiguration:
        # The configuration does not converge for this (admittely unphysical) setup, so we help a little with some viscosity
        config.params['diffusion_coef'] = 40.0
        basin_x = 1200
        basin_y = 1000
        land_x = 600
        land_y = 300
        land_site_delta = 100
        site_x = 150
        site_y = 100
        site_x_start = basin_x - land_x
        site_y_start = land_y + land_site_delta 
        config.params['turbine_x'] = 50. 
        config.params['turbine_y'] = 50. 

        for x_r in numpy.linspace(site_x_start, site_x_start + site_x, 2):
            for y_r in numpy.linspace(site_y_start, site_y_start + site_y, 2):
              turbine_pos.append((float(x_r), float(y_r)))

    else:
        border_x = config.domain.basin_x/10
        border_y = config.domain.basin_y/10

        for x_r in numpy.linspace(0.+border_x, config.domain.basin_x-border_x, 2):
            for y_r in numpy.linspace(0.+border_y, config.domain.basin_y-border_y, 2):
              turbine_pos.append((float(x_r), float(y_r)))

    config.set_turbine_pos(turbine_pos, friction=1.0)
    info_blue("Deployed " + str(len(turbine_pos)) + " turbines.")

    model = ReducedFunctional(config, scaling_factor = 10**-6)
    m0 = model.initial_control()

    p = numpy.random.rand(len(m0))
    seed = 0.1
    minconv = helpers.test_gradient_array(model.j, model.dj, m0, seed = seed, perturbation_direction = p, plot_file = "convergence_" + c.__name__ + ".pdf")
    if minconv < 1.9:
        info_red("The gradient taylor remainder test failed for the " + c.__name__ + " configuration.")
        sys.exit(1)
    else:
        info_green("The gradient taylor remainder test passed.")
