from opentidalfarm import *
import os


class TestDynamicTurbineControl(object):

    def test_gradient_passes_taylor_test(self, sw_nonlinear_problem_parameters):
        problem_params = sw_nonlinear_problem_parameters
        # Load domain
        path = os.path.dirname(__file__)
        meshfile = os.path.join(path, "mesh.xml")
        domain = FileDomain(meshfile)

        # Create a turbine
        turbine = BumpTurbine(diameter=50., friction=1.0,
                              controls=Controls(dynamic_friction=True))

        # Adjust some global options
        options["output_turbine_power"] = False


        # Create Tidalfarm
        basin_x = 640.
        basin_y = 320.
        site_x = 320.
        site_y = 160.
        site_x_start = basin_x - site_x/2
        site_y_start = basin_y - site_y/2
        turbine_pos = []
        farm = RectangularFarm(domain,
                               site_x_start=site_x_start,
                               site_x_end=site_x_start+site_x,
                               site_y_start=site_y_start,
                               site_y_end=site_y_start+site_y,
                               turbine=turbine)

        farm.add_regular_turbine_layout(num_x=2, num_y=2)

        # Set problem parameters
        problem_params.finish_time = problem_params.start_time + \
                                     2 * problem_params.dt

        friction = farm._parameters["friction"]
        farm._parameters["friction"] = [friction]*3

        # Boundary conditions
        bcs = BoundaryConditionSet()
        period = 12. * 60 * 60
        eta0 = 2.0
        k = Constant(2 * pi / (period * sqrt(problem_params.g * \
                     problem_params.depth)))
        expression = Expression(
            ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"),
            eta0=eta0, g=problem_params.g,
            depth=problem_params.depth,
            t=problem_params.start_time, k=k)

        bcs.add_bc("u", expression, 1, "weak_dirichlet")
        bcs.add_bc("u", expression, 2, "weak_dirichlet")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")

        problem_params.bcs = bcs
        problem_params.domain = domain
        problem_params.tidal_farm = farm

        # Create problem
        problem = SWProblem(problem_params)

        solver_params = CoupledSWSolver.default_parameters()
        solver_params.dump_period = -1
        solver_params.cache_forward_state = False
        solver = CoupledSWSolver(problem, solver_params)

        functional = PowerFunctional(problem)
        control = TurbineFarmControl(farm)
        rf_params = ReducedFunctionalParameters()
        rf_params.scale = 10**-6
        rf_params.automatic_scaling = False
        rf = ReducedFunctional(functional, control, solver, rf_params)
        m0 = farm.control_array

        rf(m0)

        p = numpy.random.rand(len(m0))
        seed = 0.1
        minconv = helpers.test_gradient_array(rf.__call__,
                                              rf.derivative,
                                              m0,
                                              seed=seed,
                                              perturbation_direction=p)
        assert minconv > 1.9
