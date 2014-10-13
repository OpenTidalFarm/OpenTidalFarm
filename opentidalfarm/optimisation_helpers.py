import os.path
import numpy
import dolfin
from dolfin import Constant, log, INFO
from helpers import function_eval
from dolfin_adjoint import InequalityConstraint, EqualityConstraint


def position_constraints(config):
    ''' This function returns the constraints to ensure that the turbine
    positions remain inside the domain. '''

    n = len(config.params["position"])
    if n == 0:
          raise ValueError("You need to deploy the turbines before computing the position constraints.")

    lb_x = config.domain.site_x_start + config.params["turbine_x"] / 2
    lb_y = config.domain.site_y_start + config.params["turbine_y"] / 2
    ub_x = config.domain.site_x_end - config.params["turbine_x"] / 2
    ub_y = config.domain.site_y_end - config.params["turbine_y"] / 2

    if not lb_x < ub_x or not lb_y < ub_y:
        raise ValueError("Lower bound is larger than upper bound. Is your domain large enough?")

    # The control variable is ordered as [t1_x, t1_y, t2_x, t2_y, t3_x, ...]
    lb = n * [Constant(lb_x), Constant(lb_y)]
    ub = n * [Constant(ub_x), Constant(ub_y)]
    return lb, ub


def friction_constraints(config, lb=0.0, ub=None):
    ''' This function returns the constraints to ensure that the turbine
    friction controls remain reasonable. '''

    if ub is not None and not lb < ub:
        raise ValueError("Lower bound is larger than upper bound")

    if ub is None:
        ub = 10 ** 12

    n = len(config.params["position"])
    return n * [Constant(lb)], n * [Constant(ub)]


class DomainRestrictionConstraints(InequalityConstraint):
    def __init__(self, config, feasible_area, attraction_center):
        '''
           Generates the inequality constraints to enforce the turbines in the feasible area.
           If the turbine is outside the domain, the constraints is equal to the distance between the turbine and the attraction center.
        '''
        self.config = config
        self.feasible_area = feasible_area

        # Compute the gradient of the feasible area
        fs = dolfin.FunctionSpace(feasible_area.function_space().mesh(),
                                  "DG",
                                  feasible_area.function_space().ufl_element().degree() - 1)

        feasible_area_grad = (dolfin.Function(fs),
                              dolfin.Function(fs))
        t = dolfin.TestFunction(fs)
        log(INFO, "Solving for gradient of feasible area")
        for i in range(2):
            form = dolfin.inner(feasible_area_grad[i], t) * dolfin.dx - dolfin.inner(feasible_area.dx(i), t) * dolfin.dx
            if dolfin.NonlinearVariationalSolver.default_parameters().has_parameter("linear_solver"):
                dolfin.solve(form == 0, feasible_area_grad[i], solver_parameters={"linear_solver": "cg", "preconditioner": "amg"})
            else:
                dolfin.solve(form == 0, feasible_area_grad[i], solver_parameters={"newton_solver": {"linear_solver": "cg", "preconditioner": "amg"}})
        self.feasible_area_grad = feasible_area_grad

        self.attraction_center = attraction_center

    def length(self):
        m_pos = self.config.params['turbine_pos']
        return len(m_pos)

    def function(self, m):
        ieqcons = []
        if len(self.config.params['controls']) == 2:
        # If the controls consists of the the friction and the positions, then we need to first extract the position part
            assert(len(m) % 3 == 0)
            m_pos = m[len(m) / 3:]
        else:
            m_pos = m

        for i in range(len(m_pos) / 2):
            x = m_pos[2 * i]
            y = m_pos[2 * i + 1]
            try:
                ieqcons.append(function_eval(self.feasible_area, (x, y)))
            except RuntimeError:
                print "Warning: a turbine is outside the domain"
                ieqcons.append((x - self.attraction_center[0]) ** 2 + (y - self.attraction_center[1]) ** 2)  # Point is outside domain

        arr = -numpy.array(ieqcons)
        if any(arr <= 0):
          log(INFO, "Domain restriction inequality constraints (should be >= 0): %s" % arr)
        return arr

    def jacobian(self, m):
        ieqcons = []
        if len(self.config.params['controls']) == 2:
        # If the controls consists of the the friction and the positions, then we need to first extract the position part
            assert(len(m) % 3 == 0)
            m_pos = m[len(m) / 3:]
        else:
            m_pos = m

        for i in range(len(m_pos) / 2):
            x = m_pos[2 * i]
            y = m_pos[2 * i + 1]
            primes = numpy.zeros(len(m))
            try:
                primes[2 * i] = function_eval(self.feasible_area_grad[0], (x, y))
                primes[2 * i + 1] = function_eval(self.feasible_area_grad[1], (x, y))

            except RuntimeError:
                primes[2 * i] = 2 * (x - self.attraction_center[0])
                primes[2 * i + 1] = 2 * (y - self.attraction_center[1])
            ieqcons.append(primes)

        return -numpy.array(ieqcons)

def get_domain_constraints(config, feasible_area, attraction_center):
    return DomainRestrictionConstraints(config, feasible_area, attraction_center)

def get_distance_function(config, domains):
    V = dolfin.FunctionSpace(config.domain.mesh, "CG", 1)
    v = dolfin.TestFunction(V)
    d = dolfin.TrialFunction(V)
    sol = dolfin.Function(V)
    s = dolfin.interpolate(Constant(1.0), V)
    domains_func = dolfin.Function(dolfin.FunctionSpace(config.domain.mesh, "DG", 0))
    domains_func.vector().set_local(domains.array().astype(numpy.float))

    def boundary(x):
        eps_x = config.params["turbine_x"]
        eps_y = config.params["turbine_y"]

        min_val = 1
        for e_x, e_y in [(-eps_x, 0), (eps_x, 0), (0, -eps_y), (0, eps_y)]:
            try:
                min_val = min(min_val, domains_func((x[0] + e_x, x[1] + e_y)))
            except RuntimeError:
                pass

        return min_val == 1.0

    bc = dolfin.DirichletBC(V, 0.0, boundary)

    # Solve the diffusion problem with a constant source term
    log(INFO, "Solving diffusion problem to identify feasible area ...")
    a = dolfin.inner(dolfin.grad(d), dolfin.grad(v)) * dolfin.dx
    L = dolfin.inner(s, v) * dolfin.dx
    dolfin.solve(a == L, sol, bc)

    return sol

class ConvexPolygonSiteConstraint(InequalityConstraint):
    ''' Generates the inequality constraints for generic polygon constraints.
    The parameter polygon must be a list of point coordinates that describes the
    site edges in anti-clockwise order. '''

    def __init__(self, farm, vertices):
        self.farm = farm

        V = numpy.array(vertices)
        assert len(V.shape) == 2
        assert V.shape[1] == 2

        nvertices = V.shape[0]

        # algorithm taken from vert2con.m
        c = numpy.sum(V, axis=0)/float(nvertices)
        V = V - numpy.tile(c, (nvertices, 1))
        A = numpy.nan * numpy.zeros((nvertices, 2))
        for i in range(nvertices):
            j = (i + 1) % nvertices
            F = V[[i, j], :]
            ones = numpy.ones(2)
            A[i, :] = numpy.linalg.solve(F, ones)

        b = numpy.ones(nvertices) + numpy.dot(A, c)

        self.A = A
        self.b = b
        self.nvertices = nvertices

        # The region inside is defined by Ax <= b.
        # So, to make something that should be >= 0,
        # our function is b - A*x.

    def length(self):
        return len(self.config.params['turbine_pos']) * self.nvertices

    def output_workspace(self):
        return numpy.array([0]*self.length())

    def function(self, m):
        ieqcons = []
        controlled_by = self.farm.turbine_specification.controls
        if (controlled_by.position and
            (controlled_by.friction or controlled_by.dynamic_friction)):
        # If the controls consists of the the friction and the positions, then we need to first extract the position part
            assert(len(m) % 3 == 0)
            m_pos = m[len(m) / 3:]
        else:
            m_pos = m

        for i in range(len(m_pos) / 2):
            pos = numpy.array([m_pos[2 * i], m_pos[2 * i + 1]])
            c = self.b - numpy.dot(self.A, pos)
            ieqcons = ieqcons + list(c)

        arr = numpy.array(ieqcons)
        if any(arr < 0):
          log(INFO, "Convex site position constraints (should be >= 0): %s" % arr)
        return arr

    def jacobian(self, m):
        ieqcons = []
        controlled_by = self.farm.turbine_specification.controls
        if (controlled_by.position and
            (controlled_by.friction or controlled_by.dynamic_friction)):
            # If the controls consists of the the friction and the positions, then we need to first extract the position part
            assert(len(m) % 3 == 0)
            m_pos = m[len(m) / 3:]
            mf_len = len(m_pos) / 2
        else:
            m_pos = m
            mf_len = 0

        for i in range(len(m_pos) / 2):
            pos = numpy.array([m_pos[2 * i], m_pos[2 * i + 1]])
            for constraint in range(self.nvertices):
                d = -self.A[constraint, :]
                prime = numpy.zeros(len(m))
                prime[mf_len + 2 * i] = d[0]
                prime[mf_len + 2 * i + 1] = d[1]
                ieqcons.append(prime)

        arr = numpy.array(ieqcons)
        return arr
