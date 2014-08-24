''' Tests the spatial order of convergence with weakly imposed
    Dirichlet conditions. '''

import sys
import math
from opentidalfarm import *
from opentidalfarm.initial_conditions import SinusoidalInitialCondition
from dolfin_adjoint import adj_reset
from dolfin import log, INFO, ERROR


class TestWeakDirichletBoundaryConditions(object):
    
    def error(self, config, eta0, k):
        state = Function(config.function_space)
        ic_expr = SinusoidalInitialCondition(config, eta0, k, 
                                             config.params["depth"])
        ic = project(ic_expr, state.function_space())
        state.assign(ic, annotate=False)

        adj_reset()
        solver = ShallowWaterSolver(config)
        solver.solve(state, annotate=False)

        analytic_sol = Expression(
            ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0",
             "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"),
            eta0=eta0, g=config.params["g"],
            depth=config.params["depth"],
            t=config.params["current_time"], k=k)
        return errornorm(analytic_sol, state)


    def compute_spatial_error(self, refinement_level):
        nx = 2**refinement_level
        ny = 1
        config = configuration.DefaultConfiguration(
            nx, ny, finite_element=finite_elements.p1dgp2) 
        domain = domains.RectangularDomain(3000, 1000, nx, ny)
        config.set_domain(domain)
        eta0 = 2.0
        k = pi/config.domain.basin_x
        config.params["finish_time"] = pi / (sqrt(config.params["g"] *
                                             config.params["depth"]) * k) / 20
        config.params["dt"] = config.params["finish_time"] / 10
        config.params["dump_period"] = -1
        config.params["output_turbine_power"] = False
        config.params["bctype"] = "dirichlet"
        config.params["u_weak_dirichlet_bc_expr"] = Expression(
            ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), 
            eta0=eta0,
            g=config.params["g"],
            depth=config.params["depth"],
            t=config.params["current_time"],
            k=k)

        return self.error(config, eta0, k)

    def compute_temporal_error(self, refinement_level):
        nx = 2**3
        ny = 1
        config = configuration.DefaultConfiguration(
            nx, ny, finite_element=finite_elements.p1dgp2) 
        domain = domains.RectangularDomain(3000, 1000, nx, ny)
        config.set_domain(domain)
        eta0 = 2.0
        k = pi/config.domain.basin_x
        config.params["finish_time"] = 2 * pi / (sqrt(config.params["g"] * 
                                                 config.params["depth"]) * k)
        config.params["dt"] = config.params["finish_time"]/(4*2**refinement_level)
        config.params["theta"] = 0.5
        config.params["dump_period"] = -1
        config.params["output_turbine_power"] = False
        config.params["bctype"] = "dirichlet"
        config.params["u_weak_dirichlet_bc_expr"] = Expression(
            ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), 
            eta0=eta0,
            g=config.params["g"],
            depth=config.params["depth"],
            t=config.params["current_time"],
            k=k)

        return self.error(config, eta0, k)

    def test_spatial_convergence_is_two(self):
        errors = []
        tests = 4
        for refinement_level in range(tests):
            errors.append(self.compute_spatial_error(refinement_level))
        # Compute the order of convergence 
        conv = [] 
        for i in range(len(errors)-1):
            conv.append(abs(math.log(errors[i+1] / errors[i], 2)))

        log(INFO, "Spatial order of convergence (expecting 2.0): %s" % str(conv))
        assert min(conv) > 1.8

    def test_temporal_convergence_is_two(self):
        errors = []
        tests = 4
        for refinement_level in range(tests):
            errors.append(self.compute_temporal_error(refinement_level))
        # Compute the order of convergence 
        conv = [] 
        for i in range(len(errors)-1):
            conv.append(abs(math.log(errors[i+1] / errors[i], 2)))

        log(INFO, "Temporal order of convergence (expecting 2.0): %s" % str(conv))
        assert min(conv) > 1.8
