import math
from opentidalfarm import *
from opentidalfarm.initial_conditions import SinusoidalInitialCondition
from dolfin_adjoint import adj_reset
from dolfin import log, INFO, ERROR


class TestFlatherBoundaryConditionsWithViscosity(object):

    def error(self, problem, eta0, k):

        # The analytical veclocity of the shallow water equations has been
        # multiplied by depth to account for the change of variable (\tilde u =
        # depth u) in this code.
        u_exact = "eta0*sqrt(g/depth) * cos(k*x[0]-sqrt(g*depth)*k*t)"
        ddu_exact = "(viscosity * eta0*sqrt(g/depth) * \
                     cos(k*x[0]-sqrt(g*depth)*k*t) * k*k)"
        eta_exact = "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"

        # The source term
        source = Expression((ddu_exact,
                            "0.0"),
                            eta0=eta0, g=problem.parameters.g,
                            depth=problem.parameters.depth,
                            t=problem.parameters.start_time,
                            k=k, viscosity=problem.parameters.viscosity)

        adj_reset()
        parameters = SWSolver.default_parameters()
        parameters.dump_period = -1
        solver = SWSolver(problem, parameters)
        for sol in solver.solve(annotate=False, u_source=source):
            pass
        state = sol["state"]

        analytic_sol = Expression((u_exact,
                                  "0",
                                  eta_exact),
                                  eta0=eta0, g=problem.parameters.g,
                                  depth=problem.parameters.depth,
                                  t=problem.parameters.finish_time,
                                  k=k)
        return errornorm(analytic_sol, state)


    def compute_spatial_error(self, linear_problem_params, refinement_level):
        nx = 2 * 2**refinement_level
        ny = 1

        linear_problem_params.domain = RectangularDomain(0, 0, 3000, 1000, nx, ny)

        eta0 = 2.0
        k = pi/3000.
        linear_problem_params.start_time = Constant(0.0)
        linear_problem_params.finish_time = Constant(pi/(sqrt(linear_problem_params.g *
                                         linear_problem_params.depth) * k) / 1000)
        linear_problem_params.dt = linear_problem_params.finish_time / 2
        linear_problem_params.include_viscosity = True
        linear_problem_params.viscosity = 10.0

        bcs = BoundaryConditionSet()
        bc_expr = Expression(
            ("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"),
            eta0=eta0,
            g=linear_problem_params.g,
            depth=linear_problem_params.depth,
            t=linear_problem_params.start_time,
            k=k
        )
        bcs.add_bc("u", bc_expr, [1, 2], "flater")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")

        # Initial condition
        ic_expr = SinusoidalInitialCondition(eta0, k,
                                             linear_problem_params.depth, 
                                             linear_problem_params.start_time)
        linear_problem_params.initial_condition = ic_expr

        problem = SWProblem(linear_problem_params)

        return self.error(problem, eta0, k)

    def test_spatial_convergence_is_two(self, sw_linear_problem_parameters):
        errors = []
        tests = 4
        for refinement_level in range(tests):
            errors.append(self.compute_spatial_error(sw_linear_problem_parameters,
                                                     refinement_level))
        # Compute the order of convergence
        conv = []
        for i in range(len(errors)-1):
            conv.append(-math.log(errors[i+1]/errors[i], 2))

        log(INFO, "Errors: %s" % errors)
        log(INFO, "Spatial order of convergence (expecting 2.0): %s" % conv)
        assert min(conv) > 1.8
