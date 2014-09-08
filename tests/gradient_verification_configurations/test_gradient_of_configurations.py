import os
import pytest
from opentidalfarm import *
set_log_level(INFO)


class TestConfigurations(object):

    def create_steady_sw_problem(self, problem_params):
        basin_x = 640
        basin_y = 320

        # Domain
        path = os.path.dirname(__file__)
        meshfile = os.path.join(path, "mesh.xml")
        domain = FileDomain(meshfile)
        problem_params.domain = domain

        # Boundary conditions
        bcs = BoundaryConditionSet()
        bcs.add_bc("u", Constant((2.0 + 1e-10, 0)), 1, "strong_dirichlet")
        bcs.add_bc("eta", Constant(0.0), 2, "strong_dirichlet")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")
        problem_params.bcs = bcs

        # Create a farm and deploy some turbines
        farm = TidalFarm(domain)
        turbine_pos = [] 
        # The configuration does not converge for this (admittely unphysical) 
        # setup, so we help a little with some viscosity
        problem_params.viscosity = 40.0
        site_x = 320.
        site_y = 160.
        site_x_start = basin_x - site_x/2
        site_y_start = basin_y - site_y/2
        farm.params['turbine_x'] = 50. 
        farm.params['turbine_y'] = 50. 
        farm.params['controls'] = ['turbine_pos']
        for x_r in numpy.linspace(site_x_start, site_x_start + site_x, 2):
            for y_r in numpy.linspace(site_y_start, site_y_start + site_y, 2):
              turbine_pos.append((float(x_r), float(y_r)))
        farm.params["turbine_x"] = 20.
        farm.params["turbine_y"] = 20.
        farm.set_turbine_pos(turbine_pos, friction=1.0)
        farm.params["output_turbine_power"] = False

        problem_params.tidal_farm = farm

        # Create the shallow water problem
        return SteadySWProblem(problem_params)

    def create_sw_problem(self, problem_params):
        basin_x = 500.
        basin_y = 500.

        # Domain
        domain = RectangularDomain(0, 0, basin_x, basin_y, 5, 5)
        problem_params.domain = domain

        # Temporal settings
        problem_params.finish_time = problem_params.start_time + \
                                     2*problem_params.dt

        # Boundary conditions
        bcs = BoundaryConditionSet()
        flather_bc_expr = Expression((
                         "2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"), 
                         eta0=2., 
                         g=problem_params.g, 
                         depth=problem_params.depth, 
                         t=problem_params.start_time, 
                         k=pi / 3000)
        bcs.add_bc("u", flather_bc_expr, [1, 2], "flather")
        bcs.add_bc("u", Constant((1e-10, 1e-10)), 3, "weak_dirichlet")
        problem_params.bcs = bcs

        problem_params.initial_condition = Constant((1e-9, 0, 0))

        border_x = basin_x/10
        border_y = basin_y/10

        # Create a farm and deploy some turbines
        farm = TidalFarm(domain)
        turbine_pos = [] 
        for x_r in numpy.linspace(border_x, basin_x - border_x, 2):
            for y_r in numpy.linspace(border_y, basin_y - border_y, 2):
              turbine_pos.append((float(x_r), float(y_r)))
        farm.params["turbine_x"] = 20.
        farm.params["turbine_y"] = 5.
        farm.params['controls'] = ['turbine_pos', 'turbine_friction']

        farm.set_turbine_pos(turbine_pos, friction=1.0)

        farm.params["output_turbine_power"] = False

        problem_params.tidal_farm = farm

        # Create the shallow water problem
        return SWProblem(problem_params)

    @pytest.mark.parametrize(("steady"), [False, True])
    def test_gradient_passes_taylor_test(self, steady, 
            sw_linear_problem_parameters, steady_sw_problem_parameters):

        # Define the discrete domain
        if steady:
            problem = self.create_steady_sw_problem(steady_sw_problem_parameters)
        else:
            problem = self.create_sw_problem(sw_linear_problem_parameters)

        # Create shallow water solver
        solver_params = CoupledSWSolver.default_parameters()
        solver_params.dump_period = -1
        solver_params.cache_forward_state = True
        solver = CoupledSWSolver(problem, solver_params)

        functional = PowerFunctional
        control = problem.parameters.tidal_farm
        rf_params = ReducedFunctionalParameters()
        rf_params.scale = 10**-6
        rf_params.automatic_scaling = False
        model = ReducedFunctional(functional, solver, rf_params)
        m0 = model.initial_control()

        p = numpy.random.rand(len(m0))
        seed = 0.1
        minconv = helpers.test_gradient_array(model.j, model.dj, m0, 
                seed=seed, perturbation_direction=p)
        assert minconv > 1.85
