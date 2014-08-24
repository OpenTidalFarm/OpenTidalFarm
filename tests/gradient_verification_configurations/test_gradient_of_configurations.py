from opentidalfarm import *
from dolfin import log, INFO, ERROR
import pytest


class TestConfigurations(object):

    @pytest.mark.parametrize(("c"), [DefaultConfiguration, SteadyConfiguration])
    def test_gradient_passes_taylor_test(self, c):

        # Define the discrete domain
        if c == SteadyConfiguration:
            path = os.path.dirname(__file__)
            meshfile = os.path.join(path, "mesh.xml")
            config = c(meshfile, inflow_direction = [1, 1])
        else:
            config = c(nx=15, ny=15)
            config.set_domain(domains.RectangularDomain(500, 500, 
                5, 5))

        config.params['dump_period'] = -1
        config.params['output_turbine_power'] = False
        config.params['finish_time'] = config.params["start_time"] + \
                                       2*config.params["dt"]

        config.params["flather_bc_expr"] = Expression((
                         "2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"), 
                         eta0=2., 
                         g=config.params["g"], 
                         depth=config.params["depth"], 
                         t=config.params["current_time"], 
                         k=pi / 3000)

        # Deploy some turbines 
        turbine_pos = [] 
        if c == SteadyConfiguration:
            # The configuration does not converge for this (admittely unphysical) 
            # setup, so we help a little with some viscosity
            config.params['viscosity'] = 40.0
            basin_x = 640
            basin_y = 320
            site_x = 320
            site_y = 160
            site_x_start = basin_x - site_x/2
            site_y_start = basin_y - site_y/2
            config.params['turbine_x'] = 50. 
            config.params['turbine_y'] = 50. 

            for x_r in numpy.linspace(site_x_start, site_x_start + site_x, 2):
                for y_r in numpy.linspace(site_y_start, site_y_start + site_y, 2):
                  turbine_pos.append((float(x_r), float(y_r)))

        else:
            border_x = config.domain.basin_x/10
            border_y = config.domain.basin_y/10
            basin_x = config.domain.basin_x
            basin_y = config.domain.basin_y

            for x_r in numpy.linspace(border_x, basin_x - border_x, 2):
                for y_r in numpy.linspace(border_y, basin_y - border_y, 2):
                  turbine_pos.append((float(x_r), float(y_r)))

        config.set_turbine_pos(turbine_pos, friction=1.0)
        log(INFO, "Deployed " + str(len(turbine_pos)) + " turbines.")

        solver = ShallowWaterSolver(config)

        model = ReducedFunctional(config, solver, scale=10**-6)
        m0 = model.initial_control()

        p = numpy.random.rand(len(m0))
        seed = 0.1
        minconv = helpers.test_gradient_array(model.j, model.dj, m0, 
                seed=seed, perturbation_direction=p)
        assert minconv > 1.85
