from opentidalfarm import *
from dolfin import log, INFO
set_log_level(INFO)
import pytest


class TestConfigurations(object):

    @pytest.mark.parametrize(("c"), [DefaultConfiguration, SteadyConfiguration])
    def test_gradient_passes_taylor_test(self, c,
            sw_linear_problem_parameters, steady_sw_problem_parameters):

        # Define the discrete domain
        if c == SteadyConfiguration:
            path = os.path.dirname(__file__)
            meshfile = os.path.join(path, "mesh.xml")
            domain = FileDomain(meshfile)

            config = c(domain)
            problem_params = steady_sw_problem_parameters

            # Domain
            problem_params.domain = domain

            # Boundary conditions
            bcs = BoundaryConditionSet()
            bcs.add_bc("u", Constant((2.0 + 1e-10, 0)), 1, "strong_dirichlet")
            bcs.add_bc("eta", Constant(2.0), 2, "strong_dirichlet")
            problem_params.bcs = bcs

        else:
            domain = RectangularDomain(500, 500, 5, 5)

            config = c(domain)

            problem_params = sw_linear_problem_parameters
            problem_params.finish_time = problem_params.start_time + \
                                           2*problem_params.dt

            # Domain
            problem_params.domain = domain

            # Boundary conditions
            bcs = BoundaryConditionSet()
            flather_bc_expr = Expression((
                             "2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"), 
                             eta0=2., 
                             g=problem_params.g, 
                             depth=problem_params.depth, 
                             t=problem_params.current_time, 
                             k=pi / 3000)
            bcs.add_bc("u", flather_bc_expr, [1, 2], "flater")
            bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")

        # Deploy some turbines 
        turbine_pos = [] 
        if c == SteadyConfiguration:
            # The configuration does not converge for this (admittely unphysical) 
            # setup, so we help a little with some viscosity
            steady_sw_problem_parameters.viscosity = 40.0
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


            problem = SteadyShallowWaterProblem(problem_params)

        else:
            border_x = config.domain.basin_x/10
            border_y = config.domain.basin_y/10
            basin_x = config.domain.basin_x
            basin_y = config.domain.basin_y

            for x_r in numpy.linspace(border_x, basin_x - border_x, 2):
                for y_r in numpy.linspace(border_y, basin_y - border_y, 2):
                  turbine_pos.append((float(x_r), float(y_r)))

            problem = ShallowWaterProblem(problem_params)


        config.set_turbine_pos(turbine_pos, friction=1.0)
        log(INFO, "Deployed " + str(len(turbine_pos)) + " turbines.")

        config.params["output_turbine_power"] = False

        solver_params = ShallowWaterSolver.default_parameters()
        solver_params.dump_period = -1
        solver_params.cache_forward_state = True

        solver = ShallowWaterSolver(problem, solver_params, config)

        model = ReducedFunctional(config, solver, scale=10**-6)
        m0 = model.initial_control()

        p = numpy.random.rand(len(m0))
        seed = 0.1
        minconv = helpers.test_gradient_array(model.j, model.dj, m0, 
                seed=seed, perturbation_direction=p)
        assert minconv > 1.85
