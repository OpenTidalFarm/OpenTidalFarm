from opentidalfarm import *
import pytest


class TestMultiSteadyState(object):

    @pytest.mark.parametrize(("steps"), [1, 3])
    def test_gradient_passes_taylor_test(self, steps, 
            multi_steady_sw_problem_parameters):

        # Some domain information extracted from the geo file
        basin_x = 640.
        basin_y = 320.

        path = os.path.dirname(__file__)
        meshfile = os.path.join(path, "mesh_coarse.xml")
        domain = FileDomain(meshfile)

        # Change the parameters such that in fact two steady state problems are solved consecutively
        problem_params = multi_steady_sw_problem_parameters
        problem_params.start_time = Constant(0.)
        problem_params.dt = Constant(1.)
        problem_params.finish_time = Constant(steps * problem_params.dt)
        problem_params.viscosity = Constant(16)
        k = Constant(pi/basin_x)

        # Domain
        problem_params.domain = domain
        problem_params.initial_condition = ConstantFlowInitialCondition([1, 1, 1])

        # Work out the expected delta eta for a free-stream of 2.5 m/s (without turbines) 
        # by assuming balance between the pressure and friction terms
        u_free_stream = 2.5
        log(INFO, "Target free-stream velocity (without turbines): %s" % u_free_stream)
        delta_eta = problem_params.friction/problem_params.depth/problem_params.g
        delta_eta *= u_free_stream**2
        delta_eta *= basin_x
        delta_eta = float(delta_eta)
        log(INFO, "Derived head-loss difference to achieve target free-stream: %s" % delta_eta)

        # Set Boundary conditions
        bcs = BoundaryConditionSet()
        expl = Expression("-delta_eta/2*cos(pi/steps*(t-1))",
                delta_eta=delta_eta, t=Constant(0), steps=steps)
        expr = Expression("delta_eta/2*cos(pi/steps*(t-1))",
                delta_eta=delta_eta, t=Constant(0), steps=steps)
        bcs.add_bc("eta", expl, 1, "strong_dirichlet")
        bcs.add_bc("eta", expr, 2, "strong_dirichlet")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")
        problem_params.bcs = bcs

        problem = MultiSteadyShallowWaterProblem(problem_params)

        # Place some turbines 
        config = UnsteadyConfiguration(domain)
        site_x = 320.
        site_y = 160.
        site_x_start = (basin_x - site_x)/2 
        site_y_start = (basin_y - site_y)/2 
        config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
        deploy_turbines(config, nx=8, ny=4)
        config.params["output_turbine_power"] = False

        solver_params = ShallowWaterSolver.default_parameters()
        solver_params.cache_forward_state = True
        solver_params.dump_period = -1
        solver_params.dolfin_solver = {"newton_solver": {"relative_tolerance": 1e-15}}
        solver = ShallowWaterSolver(problem, solver_params, config)

        rf = ReducedFunctional(config, solver)
        m0 = rf.initial_control()
        p = numpy.random.rand(len(m0))
        seed = 0.1
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=seed, perturbation_direction=p)

        assert minconv > 1.9
