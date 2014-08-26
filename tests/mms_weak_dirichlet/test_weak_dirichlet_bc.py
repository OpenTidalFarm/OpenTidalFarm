''' Tests the spatial order of convergence with weakly imposed
    Dirichlet conditions. '''

import sys
import math
from opentidalfarm import *
from opentidalfarm.initial_conditions import SinusoidalInitialCondition
from dolfin_adjoint import adj_reset
from dolfin import log, INFO, ERROR


class TestWeakDirichletBoundaryConditions(object):
    
    def error(self, problem, eta0, k):

        adj_reset()
        parameters = ShallowWaterSolver.default_parameters()
        parameters.dump_period = -1
        solver = ShallowWaterSolver(problem, parameters)
        state = solver.solve(annotate=False)

        analytic_sol = Expression(
            ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0",
             "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"),
            eta0=eta0, g=problem.parameters.g,
            depth=problem.parameters.depth,
            t=problem.parameters.current_time, k=k)
        return errornorm(analytic_sol, state)


    def compute_spatial_error(self, problem_params, refinement_level):
        nx = 2**refinement_level
        ny = 1

        # Set domain
        problem_params.domain = RectangularDomain(0, 0, 3000, 1000, nx, ny)

        eta0 = 2.0
        k = pi/3000.

        # Finite element
        problem_params.finite_element = finite_elements.p1dgp2

        # Time settings
        problem_params.finish_time = pi / (sqrt(problem_params.g *
                                     problem_params.depth) * k) / 20
        problem_params.dt = problem_params.finish_time / 10

        # Initial condition
        ic_expr = SinusoidalInitialCondition(eta0, k, 
                                             problem_params.depth,
                                             problem_params.start_time)
        problem_params.initial_condition = ic_expr


        # Boundary conditions
        bc_expr = Expression(
            ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), 
            eta0=eta0,
            g=problem_params.g,
            depth=problem_params.depth,
            t=problem_params.current_time,
            k=k)

        bcs = BoundaryConditionSet()
        bcs.add_bc("u", bc_expr, 1, "weak_dirichlet")
        bcs.add_bc("u", bc_expr, 2, "weak_dirichlet")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")
        problem_params.bcs = bcs

        problem = ShallowWaterProblem(problem_params)

        return self.error(problem, eta0, k)

    def compute_temporal_error(self, problem_params, refinement_level):
        nx = 2**3
        ny = 1
        eta0 = 2.0
        k = pi/3000.

        # Set domain
        problem_params.domain = RectangularDomain(0, 0, 3000, 1000, nx, ny)

        # Finite element
        problem_params.finite_element = finite_elements.p1dgp2

        # Time settings
        problem_params.finish_time = 2 * pi / (sqrt(problem_params.g * 
                                        problem_params.depth) * k)
        problem_params.dt = problem_params.finish_time/(4*2**refinement_level)
        problem_params.theta = 0.5

        # Initial condition
        ic_expr = SinusoidalInitialCondition(eta0, k, 
                                             problem_params.depth,
                                             problem_params.start_time)
        problem_params.initial_condition = ic_expr

        # Boundary conditions
        bcs = BoundaryConditionSet()
        bc_expr = Expression(
            ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), 
            eta0=eta0,
            g=problem_params.g,
            depth=problem_params.depth,
            t=problem_params.current_time,
            k=k)
        bcs.add_bc("u", bc_expr, 1, "weak_dirichlet")
        bcs.add_bc("u", bc_expr, 2, "weak_dirichlet")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")
        problem_params.bcs = bcs

        problem = ShallowWaterProblem(problem_params)

        return self.error(problem, eta0, k)

    def test_spatial_convergence_is_two(self, sw_linear_problem_parameters):
        errors = []
        tests = 4
        for refinement_level in range(tests):
            error = self.compute_spatial_error(sw_linear_problem_parameters,
                                               refinement_level)
            errors.append(error)
        # Compute the order of convergence 
        conv = [] 
        for i in range(len(errors)-1):
            conv.append(abs(math.log(errors[i+1] / errors[i], 2)))

        log(INFO, "Spatial Taylor remainders are : %s" % str(errors))
        log(INFO, "Spatial order of convergence (expecting 2.0): %s" % str(conv))
        assert min(conv) > 1.8

    def test_temporal_convergence_is_two(self, sw_linear_problem_parameters):
        errors = []
        tests = 4
        for refinement_level in range(tests):
            error = self.compute_temporal_error(sw_linear_problem_parameters,
                                                refinement_level)
            errors.append(error)
        # Compute the order of convergence 
        conv = [] 
        for i in range(len(errors)-1):
            conv.append(abs(math.log(errors[i+1] / errors[i], 2)))

        log(INFO, "Temporal Taylor remainders are : %s" % str(errors))
        log(INFO, "Temporal order of convergence (expecting 2.0): %s" % str(conv))
        assert min(conv) > 1.8
