''' Tests the spatial order of convergence with weakly imposed
    Dirichlet conditions. '''

import sys
import math
from opentidalfarm import *
from dolfin_adjoint import adj_reset
from dolfin import log, INFO, ERROR


class TestWeakDirichletBoundaryConditions(object):

    def error(self, problem, solution):

        adj_reset()
        parameters = CoupledSWSolver.default_parameters()
        parameters.dump_period = -1
        solver = CoupledSWSolver(problem, parameters)
        for sol in solver.solve(annotate=False):
            pass
        state = sol["state"]

        solution.t = problem.parameters.finish_time
        return errornorm(solution, state)


    def compute_spatial_error(self, problem_params, refinement_level):
        nx = 2 * 2**refinement_level
        ny = 5

        # Set domain
        problem_params.domain = RectangularDomain(0, 0, 3000, 1000, nx, ny)

        eta0 = 2.0
        k = pi/3000.

        # Finite element
        problem_params.finite_element = finite_elements.p0p1

        # Time settings
        problem_params.start_time = Constant(0.0)
        problem_params.finish_time = Constant(pi / (sqrt(problem_params.g *
                                     problem_params.depth) * k) / 1.)
        problem_params.dt = problem_params.finish_time / 20.
        problem_params.include_viscosity = False
        problem_params.include_advection = True
        problem_params.linear_divergence = True
        problem_params.theta = 0.5

        u_str = "eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)"
        dudx_str = "-eta0*sqrt(g/depth)*k*sin(k*x[0]-sqrt(g*depth)*k*t)"
        #dudx_str = "0."
        dudx2_str = "-eta0*sqrt(g/depth)*k*k*cos(k*x[0]-sqrt(g*depth)*k*t)"
        dudx2_str ="0."
        eta_str = "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"
        solution = Expression(
            (u_str, "0", eta_str),
            eta0=eta0, g=problem_params.g,
            depth=problem_params.depth,
            t=problem_params.start_time, k=k)
        problem_params.friction = 0.0
        u_source = Expression(
                ('{u}*{dudx}+friction*pow({u},2)/depth-viscosity*{dudx2}'.format(u=u_str, dudx=dudx_str, dudx2=dudx2_str),"0"),
                #('friction*pow({u},2)/depth'.format(u=u_str, dudx=dudx_str),"0"),
            eta0=eta0, g=problem_params.g,
            depth=problem_params.depth,
            t=problem_params.start_time, k=k,
            friction=problem_params.friction,
            viscosity=problem_params.viscosity)
        problem_params.f_u = u_source

        problem_params.initial_condition = solution

        # Boundary conditions
        bcs = BoundaryConditionSet()
        bc_expr = Expression(
            (u_str, "0"),
            eta0=eta0,
            g=problem_params.g,
            depth=problem_params.depth,
            t=problem_params.start_time,
            k=k)
        bcs.add_bc("u", bc_expr, 1, "weak_dirichlet")
        bcs.add_bc("u", bc_expr, 2, "weak_dirichlet")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")
        problem_params.bcs = bcs

        problem = SWProblem(problem_params)

        return self.error(problem, solution)

    def compute_temporal_error(self, problem_params, refinement_level):
        nx = 2**3
        ny = 5
        eta0 = 2.0
        k = pi/3000.

        # Set domain
        problem_params.domain = RectangularDomain(0, 0, 3000, 1000, nx, ny)

        # Finite element
        problem_params.finite_element = finite_elements.p1dgp2

        # Time settings
        problem_params.start_time = Constant(0.0)
        problem_params.finish_time = Constant(2 * pi / (sqrt(problem_params.g *
                                        problem_params.depth) * k))
        problem_params.dt = problem_params.finish_time/(4*2**refinement_level)
        problem_params.theta = 0.5
        problem_params.include_viscosity = False

        u_str = "eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)"
        dudx_str = "-eta0*sqrt(g/depth)*sin(k*x[0]-sqrt(g*depth)*k*t)"
        eta_str = "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"
        solution = Expression(
            (u_str, "0", eta_str),
            eta0=eta0, g=problem_params.g,
            depth=problem_params.depth,
            t=problem_params.start_time, k=k)

        problem_params.initial_condition = solution

        # Boundary conditions
        bcs = BoundaryConditionSet()
        bc_expr = Expression(
            (u_str, "0"),
            eta0=eta0,
            g=problem_params.g,
            depth=problem_params.depth,
            t=problem_params.start_time,
            k=k)
        bcs.add_bc("u", bc_expr, 1, "weak_dirichlet")
        bcs.add_bc("u", bc_expr, 2, "weak_dirichlet")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")
        problem_params.bcs = bcs

        problem = SWProblem(problem_params)

        return self.error(problem, solution)

    def test_spatial_convergence_is_first_order(self, sw_nonlinear_problem_parameters):
        errors = []
        tests = 4
        for refinement_level in range(tests):
            error = self.compute_spatial_error(sw_nonlinear_problem_parameters,
                                               refinement_level)
            errors.append(error)
        # Compute the order of convergence
        conv = []
        for i in range(len(errors)-1):
            conv.append(-math.log(errors[i+1] / errors[i], 2))

        log(INFO, "Spatial Taylor remainders are : %s" % str(errors))
        log(INFO, "Spatial order of convergence (expecting 1.0): %s" % str(conv))
        assert min(conv) > 1.0

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
            conv.append(-math.log(errors[i+1] / errors[i], 2))

        log(INFO, "Temporal Taylor remainders are : %s" % str(errors))
        log(INFO, "Temporal order of convergence (expecting 2.0): %s" % str(conv))
        assert min(conv) > 1.8
