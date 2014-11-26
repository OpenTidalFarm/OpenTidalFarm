''' This test checks the correctness of the gradient with the
    smeared turbine representation.
'''
from opentidalfarm import *


class TestSmearedTurbine(object):

    def test_gradient_passes_taylor_test(self, sw_linear_problem_parameters):
        parameters["form_compiler"]["quadrature_degree"] = 4
        prob_params=sw_linear_problem_parameters

        nx = 5
        ny = 5
        domain = RectangularDomain(0, 0, 3000, 1000, nx, ny)

        turbine = SmearedTurbine(friction=12.0)

        # Boundary conditions
        site_x_start = 750
        site_x = 1500
        site_y_start = 250
        site_y = 500

        farm = RectangularFarm(domain,
                               site_x_start=site_x_start,
                               site_x_end=site_x_start+site_x,
                               site_y_start=site_y_start,
                               site_y_end=site_y_start+site_y,
                               turbine=turbine)
        # Switch to a smeared turbine representation
        prob_params.tidal_farm = farm

        prob_params.domain = domain
        prob_params.initial_condition = Constant((1, 0, 0))

        prob_params.finish_time = prob_params.start_time + 3*prob_params.dt

        k = Constant(pi/site_x)
        bcs = BoundaryConditionSet()
        bc_expr = Expression(
            ("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"),
            eta0=2.,
            g=prob_params.g,
            depth=prob_params.depth,
            t=prob_params.start_time,
            k=k
        )
        bcs.add_bc("u", bc_expr, [1, 2], "flather")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")

        class Site(SubDomain):
            def inside(self, x, on_boundary):
                return (between(x[0], (site_x_start, site_x_start+site_x)) and
                        between(x[1], (site_y_start, site_y_start+site_y)))

        site = Site()
        d = CellFunction("size_t", farm.domain.mesh)
        d.set_all(0)
        site.mark(d, 1)
        farm.site_dx = Measure("dx")[d]

        problem = SWProblem(prob_params)

        solver_params = CoupledSWSolver.default_parameters()
        solver_params.dump_period = -1
        solver = CoupledSWSolver(problem, solver_params)

        functional = PowerFunctional(problem)
        control = TurbineFarmControl(farm)
        rf_params = ReducedFunctionalParameters()
        rf_params.automatic_scaling = False
        rf = ReducedFunctional(functional, control, solver, rf_params)
        # Ensure the same seed value accross all CPUs
        numpy.random.seed(33)
        m0 = numpy.random.rand(len(farm.control_array))

        seed = 0.1
        minconv = helpers.test_gradient_array(rf.__call__, rf.derivative, m0, seed=seed)

        assert minconv > 1.9
