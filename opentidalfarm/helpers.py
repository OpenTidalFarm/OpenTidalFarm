import random
import yaml
import os.path
import dolfin
import numpy
from dolfin import *
from dolfin_adjoint import *


def norm_approx(u, alpha=1e-4):
    r""" A smooth approximation to :math:`\|u\|`:

    .. math:: \|u\|_\alpha = \sqrt(u^2 + alpha^2)

    :param u: The coefficient.
    :param alpha: The approximation coefficient.
    :returns: ufl expression -- the approximate norm of u.

    """
    return sqrt(inner(u, u) + alpha**2)


def smooth_uflmin(a, b, alpha=1e-8):
    r""" A smooth approximation to :math:`\min(a, b)`:

    .. math:: \text{min}_\alpha(a, b) = \frac{1}{2} (a + b - \|a - b\|_\alpha)

    :param a: First argument to :math:`\min`.
    :param b: Second argument to :math:`\min`.
    :param alpha: The approximation coefficient.
    :returns: ufl expression -- the approximate :math:`\min` function.
    """
    return (a + b - norm_approx(a - b, alpha=alpha)) / 2


def get_rank():
    """ The processor number.

    :returns: int -- The processor number.
    """
    rank = MPI.rank(mpi_comm_world())

    return rank


def test_gradient_array(J, dJ, x, seed=0.01, perturbation_direction=None,
                        number_of_tests=5, plot_file=None):
    '''Checks the correctness of the derivative dJ.
       x must be an array that specifies at which point in the parameter space
       the gradient is to be checked. The functions J(x) and dJ(x) must return
       the functional value and the functional derivative respectivaly.

       This function returns the order of convergence of the Taylor
       series remainder, which should be 2 if the gradient is correct.'''

    # We will compute the gradient of the functional with respect to the
    # initial condition, and check its correctness with the Taylor remainder
    # convergence test.
    log(INFO, "Running Taylor remainder convergence analysis to check the "
              "gradient... ")

    # First run the problem unperturbed
    j_direct = J(x)

    # Randomise the perturbation direction:
    if perturbation_direction is None:
        perturbation_direction = x.copy()
        # Make sure that we use a consistent seed value accross all processors
        random.seed(243)
        for i in range(len(x)):
            perturbation_direction[i] = random.random()

    # Run the forward problem for various perturbed initial conditions
    functional_values = []
    perturbations = []
    perturbation_sizes = [seed / (2 ** i) for i in range(number_of_tests)]
    for perturbation_size in perturbation_sizes:
        perturbation = perturbation_direction.copy() * perturbation_size
        perturbations.append(perturbation)

        perturbed_x = x.copy() + perturbation
        functional_values.append(J(perturbed_x))

    # First-order Taylor remainders (not using adjoint)
    no_gradient = [abs(perturbed_j - j_direct) for perturbed_j in
                   functional_values]

    dj = dJ(x, forget=True)
    log(INFO, "Absolute functional evaluation differences: %s" % no_gradient)
    log(INFO, "Convergence orders for Taylor remainder without adjoint \
               information (should all be 1): %s" %
               convergence_order(no_gradient))

    with_gradient = []
    for i in range(len(perturbations)):
        remainder = abs(functional_values[i] - j_direct - numpy.dot(perturbations[i],
                                                              dj))
        with_gradient.append(remainder)

    log(INFO, "Absolute functional evaluation differences with adjoint: %s" %
               with_gradient)
    log(INFO, "Convergence orders for Taylor remainder with adjoint \
               information (should all be 2): %s" %
               convergence_order(with_gradient))

    if plot_file:
        first_order = [xx for xx in perturbation_sizes]
        second_order = [xx ** 2 for xx in perturbation_sizes]

        import pylab
        pylab.figure()
        pylab.loglog(perturbation_sizes, first_order, 'b--',
                     perturbation_sizes, second_order, 'g--',
                     perturbation_sizes, no_gradient, 'bo-',
                     perturbation_sizes, with_gradient, 'go-')
        pylab.legend(('First order convergence', 'Second order convergence',
                     'Taylor remainder without gradient',
                     'Taylor remainder with gradient'),
                     'lower right', shadow=True, fancybox=True)
        pylab.xlabel("Perturbation size")
        pylab.ylabel("Taylor remainder")
        pylab.savefig(plot_file)

    return min(convergence_order(with_gradient))


class StateWriter:
    def __init__(self, solver, callback=None):
        self.timestep = 0
        self.solver = solver
        self.optimisation_iteration = solver.optimisation_iteration
        self.u_out, self.p_out = self.output_files(
            solver.problem.parameters.finite_element.func_name)
        self.M_u_out, self.v_out, self.u_out_state = self.u_output_projector(
            solver.function_space.mesh())
        self.M_p_out, self.q_out, self.p_out_state = self.p_output_projector(
            solver.function_space.mesh())
        self.callback = callback

    def write(self, state):
        log(PROGRESS, "Projecting velocity and pressure to CG1 for visualisation")
        rhs = assemble(inner(self.v_out, state.split()[0]) * dx)
        solve(self.M_u_out, self.u_out_state.vector(), rhs, "cg", "sor",
              annotate=False)
        rhs = assemble(inner(self.q_out, state.split()[1]) * dx)
        solve(self.M_p_out, self.p_out_state.vector(), rhs, "cg", "sor",
              annotate=False)

        self.u_out << self.u_out_state
        self.p_out << self.p_out_state

        if self.callback is not None:
            self.callback(state, self.u_out_state, self.p_out_state,
                          self.timestep, self.optimisation_iteration)

        self.timestep += 1

    def u_output_projector(self, mesh):
        # Projection operator for output.
        Output_V = VectorFunctionSpace(mesh, 'CG', 1, dim=2)

        u_out = TrialFunction(Output_V)
        v_out = TestFunction(Output_V)

        M_out = assemble(inner(v_out, u_out) * dx)
        out_state = Function(Output_V)

        return M_out, v_out, out_state

    def p_output_projector(self, mesh):
        # Projection operator for output.
        Output_V = FunctionSpace(mesh, 'CG', 1)

        u_out = TrialFunction(Output_V)
        v_out = TestFunction(Output_V)

        M_out = assemble(inner(v_out, u_out) * dx)
        out_state = Function(Output_V)

        return M_out, v_out, out_state

    def output_files(self, basename):
        # Output file
        u_out = File(os.path.join(self.solver.parameters.output_dir,
            "iter_{}".format(self.optimisation_iteration), basename + "_u.pvd"),
            "compressed")
        p_out = File(os.path.join(self.solver.parameters.output_dir,
            "iter_{}".format(self.optimisation_iteration), basename + "_p.pvd"),
            "compressed")
        return u_out, p_out


def cpu0only(f):
    ''' A decorator class that only evaluates on the first CPU in a parallel
        environment. '''
    def decorator(self, *args, **kw):
        myid = get_rank()
        if myid == 0:
            f(self, *args, **kw)

    return decorator


def function_eval(func, point):
    ''' A parallel safe evaluation of dolfin functions '''
    try:
        val = func(point)
    except RuntimeError:
        val = -numpy.inf

    if dolfin.__version__ >= '1.3.0+':
        maxval = MPI.max(mpi_comm_world(), val)
    else:
        maxval = MPI.max(val)

    if maxval == -numpy.inf:
        raise RuntimeError("Point is outside the domain")
    else:
        return maxval


class FrozenClass(object):
    """ A class which can be (un-)frozen. If the class is frozen, no attributes
        can be added to the class. """

    __isfrozen = True

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class. %r is not a valid \
attribute." % (self, key))

        object.__setattr__(self, key, value)

    def _freeze(self):
        """ Disallows adding new attributes. """
        self.__isfrozen = True

    def _unfreeze(self):
        """ Allows adding new attributes. """
        self.__isfrozen = False

    def _convert_type(self, k):
        attr = getattr(self, k)

        if isinstance(attr, bool):
            return attr
        try:
            return float(attr)
        except:
            return str(attr)

    def __str__(self):
        attrs = dir(self)
        attrs_dict = {}

        for k in attrs:
            if k.startswith("_"):
                continue
            val = self._convert_type(k)
            attrs_dict[k] = val

        return yaml.dump(attrs_dict, default_flow_style=False)

class OutputWriter(object):
    """ Suite of tools to write output to disk
    """

    def __init__(self, functional):

        self.functional = functional

    def individual_turbine_power(self, solver):
        """ Print out the individual turbine's power
        """
        log(INFO, "Computing individual turbine power extraction contribution.")
        farm = solver.problem.parameters.tidal_farm
        turbine_positions = farm.turbine_positions
        individual_turbine_data = []
        for i in range(len(turbine_positions)):
            turbine_info = {}
            turbine_info['location'] = turbine_positions[i]
            turbine_info['power'] = self.functional.Jt_individual(solver.state, i)
            turbine_info['force'] = self.functional.force_individual(solver.state, i)
            turbine_info['friction'] = farm.turbine_cache._parameters['friction'][i]
            turbine_info['friction_field'] = farm.turbine_cache['turbine_field_individual'][i]
            info("Contribution of turbine %d at x=%.3f, "
                 "y=%.3f, is %.2f kW with a friction of %.2f." % (i,
                 turbine_info['location'][0], turbine_info['location'][1],
                 turbine_info['power']*0.001, turbine_info['friction']))
