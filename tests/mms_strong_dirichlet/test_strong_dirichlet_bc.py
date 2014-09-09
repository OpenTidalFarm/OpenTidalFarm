''' Tests the spatial order of convergence with strongly imposed
    Dirichlet conditions. '''

import sys
import math
from opentidalfarm import *
from dolfin_adjoint import adj_reset
from dolfin import log, INFO, ERROR

class TestStringDirichletBoundaryConditions(object):

    def error(self, problem, eta0, k):

      adj_reset()
      params = CoupledSWSolver.default_parameters()
      params.dump_period = -1
      solver = CoupledSWSolver(problem, params)
      for s in solver.solve(annotate=False):
          pass
      state = s["state"]

      analytic_sol = Expression(
             ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0", \
             "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"), \
             eta0=eta0, g=problem.parameters.g, \
             depth=problem.parameters.depth, \
             t=float(problem.parameters.finish_time), k=k)
      return errornorm(analytic_sol, state)

    def compute_spatial_error(self, problem_params, refinement_level, sin_ic):
        nx = 2 * 2**refinement_level
        ny = 1

        eta0 = 2.0
        k = pi/3000.
        problem_params.finish_time = Constant(pi / (sqrt(problem_params.g * \
                                        problem_params.depth) * k) / 20)
        problem_params.dt = Constant(problem_params.finish_time / 4)

        # Domain
        domain = RectangularDomain(0, 0, 3000, 1000, nx, ny)
        problem_params.domain = domain

        # Boundary conditions
        bcs = BoundaryConditionSet()
        bc_expr = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), 
                                eta0=eta0, 
                                g=problem_params.g, 
                                depth=problem_params.depth, 
                                t=problem_params.start_time, 
                                k=k)

        bcs.add_bc("u", bc_expr, 1)
        bcs.add_bc("u", bc_expr, 2)
        bcs.add_bc("u", bc_expr, 3)
        problem_params.bcs = bcs

        # Initial condition
        ic_expr = sin_ic(eta0, k, 
                         problem_params.depth,
                         problem_params.start_time)
        problem_params.initial_condition = ic_expr

        problem = SWProblem(problem_params)

        return self.error(problem, eta0, k)

    def compute_temporal_error(self, problem_params, refinement_level, sin_ic):
        nx = 2**3
        ny = 1
        eta0 = 2.0
        k = pi/3000.

        domain = RectangularDomain(0, 0, 3000, 1000, nx, ny)
        problem_params.domain = domain

        problem_params.finish_time = 2 * pi / (sqrt(problem_params.g * 
            problem_params.depth) * k)
        problem_params.dt = Constant(problem_params.finish_time / 
                (4 * 2**refinement_level))
        problem_params.theta = 0.5

        # Boundary conditions
        bcs = BoundaryConditionSet()
        bc_expr = Expression(
            ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), 
            eta0=eta0, 
            g=problem_params.g, 
            depth=problem_params.depth, 
            t=problem_params.start_time, 
            k=k)

        bcs.add_bc("u", bc_expr, 1, "strong_dirichlet")
        bcs.add_bc("u", bc_expr, 2, "strong_dirichlet")
        bcs.add_bc("u", bc_expr, 3, "strong_dirichlet")
        problem_params.bcs = bcs

        # Initial condition
        ic_expr = sin_ic(eta0, k, 
                         problem_params.depth,
                         problem_params.start_time)
        problem_params.initial_condition = ic_expr

        problem = SWProblem(problem_params)

        return self.error(problem, eta0, k)

    
    def test_spatial_convergence_is_two(self, sw_linear_problem_parameters,
            sin_ic):
        errors = []
        tests = 4
        for refinement_level in range(tests):
            error = self.compute_spatial_error(sw_linear_problem_parameters, 
                                               refinement_level, sin_ic)
            errors.append(error)

        # Compute the order of convergence 
        conv = [] 
        for i in range(len(errors)-1):
          conv.append(-math.log(errors[i+1] / errors[i], 2))

        log(INFO, "Spatial Taylor remainders: %s" % errors)
        log(INFO, "Spatial order of convergence (expecting 2.0): %s" % conv)
        assert min(conv) > 1.8

    def test_temporal_convergence_is_two(self, sw_linear_problem_parameters,
            sin_ic):
        errors = []
        tests = 4
        for refinement_level in range(tests):
          error = self.compute_temporal_error(sw_linear_problem_parameters, 
                                               refinement_level, sin_ic)
          errors.append(error)
        # Compute the order of convergence 
        conv = [] 
        for i in range(len(errors)-1):
          conv.append(-math.log(errors[i+1] / errors[i], 2))

        log(INFO, "Temporal Taylor remainders: %s" % errors)
        log(INFO, "Temporal order of convergence (expecting 2.0): %s" % conv)
        assert min(conv) > 1.8
