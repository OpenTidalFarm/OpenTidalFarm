''' This test checks the correctness of the gradient with the
    smeared turbine representation.
'''
from opentidalfarm import *


class TestSmearedTurbine(object):

    def test_gradient_passes_taylor_test(self, sw_linear_problem_parameters):
        parameters["form_compiler"]["quadrature_degree"] = 4

        nx = 5
        ny = 5
        domain = RectangularDomain(0, 0, 3000, 1000, nx, ny)
        config = DefaultConfiguration(domain)

        # Switch to a smeared turbine representation
        config.params["controls"] = ["turbine_friction"]
        config.params["turbine_parametrisation"] = "smeared"

        sw_linear_problem_parameters.domain = domain
        sw_linear_problem_parameters.initial_condition = Constant((1, 0, 0))

        sw_linear_problem_parameters.finish_time = sw_linear_problem_parameters.start_time + \
            3*sw_linear_problem_parameters.dt

        # Boundary conditions
        site_x_start = 750
        site_x = 1500
        site_y_start = 250
        site_y = 500

        k = Constant(pi/site_x)
        bcs = BoundaryConditionSet()
        bc_expr = Expression(
            ("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"),
            eta0=2.,
            g=sw_linear_problem_parameters.g,
            depth=sw_linear_problem_parameters.depth,
            t=sw_linear_problem_parameters.start_time,
            k=k
        )
        bcs.add_bc("u", bc_expr, [1, 2], "flather")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")

        class Site(SubDomain):
            def inside(self, x, on_boundary):
                return (between(x[0], (site_x_start, site_x_start+site_x)) and
                        between(x[1], (site_y_start, site_y_start+site_y)))

        site = Site()
        d = CellFunction("size_t", config.domain.mesh)
        d.set_all(0)
        site.mark(d, 1)
        config.site_dx = Measure("dx")[d]

        problem = SWProblem(sw_linear_problem_parameters)

        solver_params = CoupledSWSolver.default_parameters()
        solver_params.dump_period = -1
        solver = CoupledSWSolver(problem, solver_params, config)

        rf = ReducedFunctional(config, solver, automatic_scaling=False)
        # Ensure the same seed value accross all CPUs
        numpy.random.seed(33)
        m0 = numpy.random.rand(len(rf.initial_control()))

        seed = 0.1
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=seed)

        assert minconv > 1.9
