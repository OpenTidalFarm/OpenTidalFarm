import numpy
from dolfin import Constant
import dolfin
from helpers import info_blue, function_eval
import os.path


class IPOptFunction(object):
    ''' The wrapper class of the objective/constaint functions that as required by the ipopt package. '''

    def __init__(self):
        pass

    def objective(self, x):
        ''' The objective function evaluated at x. '''
        print "The objective_user function must be overloaded."

    def gradient(self, x):
        ''' The gradient of the objective function evaluated at x. '''
        print "The gradient_user function must be overloaded."

    def constraints(self, x):
        ''' The constraint functions evaluated at x. '''
        return numpy.array([])

    def jacobian(self, x):
        ''' The Jacobian of the constraint functions evaluated at x. '''
        return (numpy.array([]), numpy.array([]))


def deploy_turbines(config, nx, ny, friction=21.):
    ''' Generates an array of initial turbine positions with nx x ny turbines homonginuosly distributed over the site with the specified dimensions. '''
    turbine_pos = []
    for x_r in numpy.linspace(config.domain.site_x_start + 0.5 * config.params["turbine_x"], config.domain.site_x_end - 0.5 * config.params["turbine_x"], nx):
        for y_r in numpy.linspace(config.domain.site_y_start + 0.5 * config.params["turbine_y"], config.domain.site_y_end - 0.5 * config.params["turbine_y"], ny):
            turbine_pos.append((float(x_r), float(y_r)))
    config.set_turbine_pos(turbine_pos, friction)
    info_blue("Deployed " + str(len(turbine_pos)) + " turbines.")
    return turbine_pos


def position_constraints(config):
    ''' This function returns the constraints to ensure that the turbine positions remain inside the domain. '''

    n = len(config.params["turbine_pos"])
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
    ''' This function returns the constraints to ensure that the turbine friction controls remain reasonable. '''

    if ub is not None and not lb < ub:
        raise ValueError("Lower bound is larger than upper bound")

    if ub is None:
        ub = 10 ** 12

    n = len(config.params["turbine_pos"])
    return n * [Constant(lb)], n * [Constant(ub)]


def get_minimum_distance_constraint_func(config, min_distance=None):
    if not 'turbine_pos' in config.params['controls']:
        raise NotImplementedError("Inequality contraints for the distance only make sense if the turbine positions are control variables.")

    if not min_distance:
        min_distance = 1.5 * max(config.params["turbine_x"], config.params["turbine_y"])

    def l2norm(x):
        return sum([v ** 2 for v in x])

    def f_ieqcons(m):
        ieqcons = []
        if len(config.params['controls']) == 2:
        # If the controls consists of the the friction and the positions, then we need to first extract the position part
            assert(len(m) % 3 == 0)
            m_pos = m[len(m) / 3:]
        else:
            m_pos = m

        for i in range(len(m_pos) / 2):
            for j in range(len(m_pos) / 2):
                if i <= j:
                    continue
                ieqcons.append(l2norm([m_pos[2 * i] - m_pos[2 * j], m_pos[2 * i + 1] - m_pos[2 * j + 1]]) - min_distance ** 2)

        arr = numpy.array(ieqcons)
        if any(arr <= 0):
          info_blue("Minimum distance inequality constraints (should be > 0): %s" % arr)
        return numpy.array(ieqcons)

    def fprime_ieqcons(m):
        ieqcons = []
        if len(config.params['controls']) == 2:
            # If the controls consists of the the friction and the positions, then we need to first extract the position part
            assert(len(m) % 3 == 0)
            m_pos = m[len(m) / 3:]
            mf_len = len(m_pos) / 2
        else:
            m_pos = m
            mf_len = 0

        for i in range(len(m_pos) / 2):
            for j in range(len(m_pos) / 2):
                if i <= j:
                    continue
                prime_ieqcons = numpy.zeros(len(m))

                # The control vector contains the friction coefficients first, so we need to shift here
                prime_ieqcons[mf_len + 2 * i] = 2 * (m_pos[2 * i] - m_pos[2 * j])
                prime_ieqcons[mf_len + 2 * j] = -2 * (m_pos[2 * i] - m_pos[2 * j])
                prime_ieqcons[mf_len + 2 * i + 1] = 2 * (m_pos[2 * i + 1] - m_pos[2 * j + 1])
                prime_ieqcons[mf_len + 2 * j + 1] = -2 * (m_pos[2 * i + 1] - m_pos[2 * j + 1])

                ieqcons.append(prime_ieqcons)
        return numpy.array(ieqcons)

    return {'type': 'ineq', 'fun': f_ieqcons, 'jac': fprime_ieqcons}


def plot_site_constraints(config, vertices):

    ineqs = []
    for p in range(len(vertices)):
        # x1 and x2 are the two points that describe one of the sites edge
        x1 = numpy.array(vertices[p])
        x2 = numpy.array(vertices[(p + 1) % len(vertices)])
        c = x2 - x1
        # Normal vector of c
        n = [c[1], -c[0]]

        ineqs.append((x1, n))

    class SiteConstraintExpr(dolfin.Expression):
        def eval(self, value, x):
            inside = True
            for x1, n in ineqs:
                # The inequality for this edge is: g(x) := n^T.(x1-x) >= 0
                inside = inside and (numpy.dot(n, x1 - x) >= 0)

            value[0] = int(inside)

    f = dolfin.project(SiteConstraintExpr(), config.turbine_function_space)
    out_file = dolfin.File(config.params['base_path'] + os.path.sep + "site_constraints.pvd", "compressed")
    out_file << f


def generate_site_constraints(config, vertices, penalty_factor=1e3, slack_eps=0):
    ''' Generates the inequality constraints for generic polygon constraints. The parameter polygon
        must be a list of point coordinates that describes the site edges in anti-clockwise order.
        The argument slack_eps is used to increase or decrease the site by an epsilon value - this is useful to avoid rounding problems. '''

    # To begin with, lets save a vtu that visualises the constraints
    plot_site_constraints(config, vertices)

    def f_ieqcons(m):
        ieqcons = []
        if len(config.params['controls']) == 2:
        # If the controls consists of the the friction and the positions, then we need to first extract the position part
            assert(len(m) % 3 == 0)
            m_pos = m[len(m) / 3:]
        else:
            m_pos = m

        for i in range(len(m_pos) / 2):
            for p in range(len(vertices)):
                # x1 and x2 are the two points that describe one of the sites edge
                x1 = numpy.array(vertices[p])
                x2 = numpy.array(vertices[(p + 1) % len(vertices)])
                c = x2 - x1
                # Normal vector of c
                n = [c[1], -c[0]]

                # The inequality for this edge is: g(x) := n^T.(x1-x) >= 0
                x = m_pos[2 * i:2 * i + 2]
                ieqcons.append(penalty_factor * (numpy.dot(n, x1 - x) + slack_eps))

        return numpy.array(ieqcons)

    def fprime_ieqcons(m):
        ieqcons = []
        if len(config.params['controls']) == 2:
            # If the controls consists of the the friction and the positions, then we need to first extract the position part
            assert(len(m) % 3 == 0)
            m_pos = m[len(m) / 3:]
            mf_len = len(m_pos) / 2
        else:
            mf_len = 0
            m_pos = m

        for i in range(len(m_pos) / 2):
            for p in range(len(vertices)):
                # x1 and x2 are the two points that describe one of the sites edge
                x1 = numpy.array(vertices[p])
                x2 = numpy.array(vertices[(p + 1) % len(vertices)])
                c = x2 - x1
                # Normal vector of c
                n = [c[1], -c[0]]

                prime_ieqcons = numpy.zeros(len(m))

                # The control vector contains the friction coefficients first, so we need to shift here
                prime_ieqcons[mf_len + 2 * i] = -penalty_factor * n[0]
                prime_ieqcons[mf_len + 2 * i + 1] = -penalty_factor * n[1]

                ieqcons.append(prime_ieqcons)
        return numpy.array(ieqcons)

    return {'type': 'ineq', 'fun': f_ieqcons, 'jac': fprime_ieqcons}


class DomainRestrictionConstraints:
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
        info_blue("Solving for gradient of feasible area")
        for i in range(2):
            form = dolfin.inner(feasible_area_grad[i], t) * dolfin.dx - dolfin.inner(feasible_area.dx(i), t) * dolfin.dx
            dolfin.solve(form == 0, feasible_area_grad[i])
        self.feasible_area_grad = feasible_area_grad

        self.attraction_center = attraction_center

    def generate(self, jac=True):
        ''' Returns a dictionar with the inequality constrains.
           If :arg: jac is True, the return dictionary also contains the Jacobian function.
        '''

        def f_ieqcons(m):
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
              info_blue("Domain restriction inequality constraints (should be >= 0): %s" % arr)
            return arr

        def fprime_ieqcons(m):
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

        if jac:
            return {'type': 'ineq', 'fun': f_ieqcons, 'jac': fprime_ieqcons}
        else:
            return {'type': 'ineq', 'fun': f_ieqcons}


def get_domain_constraints(config, feasible_area, attraction_center, jac=True):
    return DomainRestrictionConstraints(config, feasible_area, attraction_center).generate(jac)


def merge_constraints(ineq1, ineq2):
    assert(ineq1['type'] == 'ineq' and ineq2['type'] == 'ineq')

    ineq_fun = lambda m: numpy.array(list(ineq1['fun'](m)) + list(ineq2['fun'](m)))
    if 'jac' in ineq1 and 'jac' in ineq2:
        prime_fun = lambda m: numpy.array(list(ineq1['jac'](m)) + list(ineq2['jac'](m)))
        return {'type': 'ineq', 'fun': ineq_fun, 'jac': prime_fun}
    else:
        return {'type': 'ineq', 'fun': ineq_fun}


def get_distance_function(config, domains):
    V = dolfin.FunctionSpace(config.domain.mesh, "CG", 1)
    v = dolfin.TestFunction(V)
    d = dolfin.Function(V)
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
    info_blue("Solving diffusion problem to identify feasible area ...")
    F = (dolfin.inner(dolfin.grad(d), dolfin.grad(v)) - dolfin.inner(s, v)) * dolfin.dx
    dolfin.solve(F == 0, d, bc)

    return d
