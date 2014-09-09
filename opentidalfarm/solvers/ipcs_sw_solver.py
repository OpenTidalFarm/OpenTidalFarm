import os.path

from dolfin import *
from dolfin_adjoint import *

from solver import Solver
from ..problems import SWProblem
from ..problems import SteadySWProblem
from ..problems import MultiSteadySWProblem
from ..helpers import StateWriter, norm_approx, smooth_uflmin, FrozenClass


class IPCSSWSolverParameters(FrozenClass):
    """ A set of parameters for a :class:`IPCSSWSolver`.

    Following parameters are available:

    :ivar dolfin_solver: The dictionary with parameters for the dolfin
        Newton solver. A list of valid entries can be printed with:

        .. code-block:: python

            info(NonlinearVariationalSolver.default_parameters(), True)

        By default, the MUMPS direct solver is used for the linear system. If
        not availabe, the default solver and preconditioner of FEniCS is used.

    :ivar dump_period: Specfies how often the solution should be dumped to disk.
        Use a negative value to disable it. Default 1.
    :ivar cache_forward_state: If True, the shallow water solutions are stored
        for every timestep and are used as initial guesses for the next solve.
        If False, the solution of the previous timestep is used as an initial guess.
        Default: True
    :ivar print_individual_turbine_power: Print out the turbine power for each
        turbine. Default: False
    :ivar quadrature_degree: The quadrature degree for the matrix assembly.
        Default: 5
    :ivar cpp_flags: A list of cpp compiler options for the code generation.
        Default: ["-O3", "-ffast-math", "-march=native"]

    """

    dolfin_solver = {"newton_solver": {}}
    dump_period = 1
    print_individual_turbine_power = False

    # Performance settings
    cache_forward_state = True
    quadrature_degree = 5
    cpp_flags = ["-O3", "-ffast-math", "-march=native"]

    def __init__(self):

        linear_solver = 'mumps' if ('mumps' in map(lambda x: x[0], linear_solver_methods())) else 'default'
        preconditioner = 'default'

        self.dolfin_solver["newton_solver"]["linear_solver"] = linear_solver
        self.dolfin_solver["newton_solver"]["preconditioner"] = preconditioner
        self.dolfin_solver["newton_solver"]["maximum_iterations"] = 20


class IPCSSWSolver(Solver):
    r"""
    This incremental pressure correction scheme (IPCS) is an operator splitting
    scheme that follows the idea of Goda [1]_.  This scheme preserves the exact
    same stability properties as Navier-Stokes and hence does not introduce
    additional dissipation in the flow.

    The :class:`IPCSSWSolver` only works with transient problems, that is with
    :class:`opentidalfarm.problems.sw.SWProblem`.

    The idea is to replace the unknown free-surface with an approximation. This
    is chosen as the free-surface solution from the previous solution.

    The time discretization is done using backward Euler, the diffusion term is
    handled with Crank-Nicholson, and the convection is handled explicitly,
    making the equations completely linear. Thus, we have a discretized version
    of the shallow water equations as

    .. math:: \frac{1}{\Delta t}\left( u^{n+1}-u^{n}
        \right)-\nabla\cdot\nu\nabla u^{n+\theta}+u^n\cdot\nabla u^{n}+g\nabla
        \eta^{n+1} &= f_u^{n+1}, \\
        \frac{1}{\Delta t}\left( \eta^{n+1}-\eta^{n} \right) + \nabla \cdot
        \left( H^{n+1} u^{n+1} \right) &= 0,

    where :math:`u^{n+\theta} = \theta{u}^{n+1}+(1-\theta)u^n, \theta =
    \frac{1}{2}`.

    For the operator splitting, we use the free-surface solution from the
    previous timestep as an estimation, giving an equation for a tentative
    velocity, :math:`\tilde{u}^{n+1}`:

    .. math:: \frac{1}{\Delta t}\left( \tilde{u}^{n+1}-u^{n}
        \right)-\nabla\cdot\nu\nabla \tilde{u}^{n+\theta}+u^n\cdot\nabla
        u^{n}+g\nabla \eta^{n}=f_u^{n+1}.

    This tenative velocity does not satisfy the divergence equation, and thus we
    define a velocity correction :math:`u^c=u^{n+1}-\tilde{u}^{n+1}`.
    Substracting the second equation from the first, we see that

    .. math::
        \frac{1}{\Delta t}u^c - \theta \nabla\cdot\nu\nabla
        u^c+g\nabla\left( \eta^{n+1} - \eta^n\right)=0, \\
        \frac{1}{\Delta t}\left( \eta^{n+1}-\eta^{n} \right) + \nabla \cdot
        \left( H^{n+1} u^c \right) = -\nabla \cdot \left( H^{n+1}
        \tilde{u}^{n+1} \right).

    The operator splitting is a first order approximation, :math:`O(\Delta t)`,
    so we can, without reducing the order of the approximation simplify the
    above to

    .. math::
        \frac{1}{\Delta t}u^c + g\nabla\left( \eta^{n+1} - \eta^n\right)=0, \\
        \frac{1}{\Delta t}\left( \eta^{n+1}-\eta^{n} \right) + \nabla \cdot
        \left( H^{n+1} u^c \right) = -\nabla \cdot \left( H^{n+1}
        \tilde{u}^{n+1} \right),

    which is reducible to the problem:

    .. math::
        \frac{1}{\Delta t}\left( \eta^{n+1}-\eta^{n} \right) - g \Delta t \nabla \cdot
        \left( H^{n+1}  \nabla\left( \eta^{n+1} - \eta^n\right) \right) = -\nabla \cdot \left( H^{n+1}
        \tilde{u}^{n+1} \right).

    The corrected velocity is then easily calculated from

    .. math::
        u^{n+1} = \tilde{u}^{n+1} - \Delta tg\nabla\left(\eta^{n+1}-\eta^n\right)

    The scheme can be summarized in the following steps:
        #. Replace the pressure with a known approximation and solve for a tenative velocity :math:`\tilde u^{n+1}`.

        #. Solve a free-surface correction equation for the free-surface, :math:`\eta^{n+1}`

        #. Use the corrected pressure to find the velocity correction and calculate :math:`u^{n+1}`

        #. Update t, and repeat.

    .. [1] Goda, Katuhiko. *A multistep technique with implicit difference schemes for calculating two-or three-dimensional cavity flows.* Journal of Computational Physics 30.1 (1979): 76-95.
    """

    def __init__(self, problem, parameters, config=None):

        if not isinstance(problem, (SWProblem, SteadySWProblem)):
            raise TypeError, "problem must be of type Problem"

        if not isinstance(parameters, IPCSSWSolverParameters):
            raise TypeError, "parameters must be of type \
IPCSSWSolverParameters."

        self.problem = problem
        self.parameters = parameters
        self.config = config

        # If cache_for_nonlinear_initial_guess is true, then we store all
        # intermediate state variables in this dictionary to be used for the
        # next solve
        self.state_cache = {}

        self.mesh = problem.parameters.domain.mesh
        self.V, self.Q = self.problem.parameters.finite_element(self.mesh)


    @staticmethod
    def default_parameters():
        return IPCSSWSolverParameters()

    def _finished(self, current_time, finish_time):
        return float(current_time - finish_time) >= - 1e3*DOLFIN_EPS

    def _generate_strong_bcs(self):

        bcs = self.problem.parameters.bcs
        facet_ids = self.problem.parameters.domain.facet_ids

        # Generate velocity boundary conditions
        bcs_u = []
        for _, expr, facet_id, _ in bcs.filter("u", "strong_dirichlet"):
            bc = DirichletBC(self.V, expr, facet_ids, facet_id)
            bcs_u.append(bc)

        # Generate free-surface boundary conditions
        bcs_eta = []
        for _, expr, facet_id, _ in bcs.filter("eta", "strong_dirichlet"):
            bc = DirichletBC(self.Q, expr, facet_ids, facet_id)
            bcs_eta.append(bc)

        return bcs_u, bcs_eta

    def solve(self, turbine_field=None, annotate=True):
        ''' Solve the shallow water equations '''

        ############################### Setting up the equations ###########################

        # Initialise solver settings
        if not type(self.problem) == SWProblem:
            raise TypeError("Do not know how to solve problem of type %s." %
                type(self.problem))

        # Get parameters
        problem_params = self.problem.parameters
        solver_params = self.parameters

        # Performance settings
        parameters['form_compiler']['quadrature_degree'] = \
            solver_params.quadrature_degree
        parameters['form_compiler']['cpp_optimize_flags'] = \
            " ".join(solver_params.cpp_flags)
        parameters['form_compiler']['cpp_optimize'] = True
        parameters['form_compiler']['optimize'] = True
        cache_forward_state = solver_params.cache_forward_state

        # Get domain measures
        ds = problem_params.domain.ds
        dx = problem_params.domain.dx

        # Get the boundary normal direction
        n = FacetNormal(self.mesh)

        # Get temporal settings
        theta = Constant(problem_params.theta)
        dt = Constant(problem_params.dt)
        finish_time = Constant(problem_params.finish_time)
        t = Constant(problem_params.start_time)

        # Get equation settings
        g = problem_params.g
        h = problem_params.depth
        include_advection = problem_params.include_advection
        include_viscosity = problem_params.include_viscosity
        nu = problem_params.viscosity
        linear_divergence = problem_params.linear_divergence

        # Get boundary conditions
        bcs = problem_params.bcs

        # Get function spaces
        V, Q = self.V, self.Q

        # Test and trial functions
        v = TestFunction(V)
        q = TestFunction(Q)
        u = TrialFunction(V)
        p = TrialFunction(Q)

        # Functions
        u0 = Function(V, name="u0")
        u1 = Function(V, name="u1")
        eta0 = Function(Q, name="eta0")
        eta1 = Function(Q, name="eta1")

        if u_source is None:
            u_source = Constant((0, 0))

        # Load initial conditions
        # Projection is necessary to obtain 2nd order convergence
        # FIXME: The problem should specify the ic for each component separately.
        u_ic = project(problem_params.initial_condition_u, self.V)
        u0.assign(u_ic, annotate=False)
        eta_ic = project(problem_params.initial_condition_eta, self.Q)
        eta0.assign(eta_ic, annotate=False)

        # Tentative velocity step
        u_mean = theta * u + (1. - theta) * u0
        u_diff = u - u0
        F_u_tent = ((1/dt) * inner(v, u_diff) * dx()
                    #+ inner(v, grad(u0)*u0) * dx()
                    + inner(grad(v), grad(u_mean)) * dx()
                    - inner(div(v), eta0) * dx()
                    + inner(v, eta0*n) * ds()
                    - inner(v, u_source) * dx())

        a_u_tent = lhs(F_u_tent)
        L_u_tent = rhs(F_u_tent)

        # Pressure correction
        a_p_corr = (q*p + dt**2 * theta**2 * h * inner(grad(q), grad(p)))*dx()
        L_p_corr = (q*eta0 + dt**2 * theta**2 * h * inner(grad(q), grad(eta0)))*dx() \
                 - dt*theta*h*q*div(u1)*dx()

        # Velocity correction
        a_u_corr = inner(v, u)*dx()
        L_u_corr = inner(v, u1)*dx() - dt*inner(v, grad(eta1-eta0))*dx()

        # Assemble matrices
        A_u_tent = assemble(a_u_tent)
        A_p_corr = assemble(a_p_corr)
        A_u_corr = assemble(a_u_corr)

        yield({"time": t,
               "u": u0,
               "eta": eta0,
               "is_final": self._finished(t, finish_time)})

        log(INFO, "Start of time loop")
        adjointer.time.start(t)
        timestep = 0

        bcu, bceta = self._generate_strong_bcs()

        while not self._finished(t, finish_time):
            # Update timestep
            timestep += 1
            t = Constant(t + dt)

            # Update bc's
            t_theta = Constant(t - (1.0 - theta) * dt)
            bcs.update_time(t, only_type=["strong_dirichlet"])
            bcs.update_time(t_theta, exclude_type=["strong_dirichlet"])

            # Update source term
            if u_source is not None:
                u_source.t = Constant(t_theta)

            # Compute tentative velocity step
            b = assemble(L_u_tent)
            for bc in bcu: bc.apply(A_u_tent, b)

            solve(A_u_tent, u1.vector(), b)

            # Pressure correction
            b = assemble(L_p_corr)
            for bc in bceta: bc.apply(A_p_corr, b)

            solve(A_p_corr, eta1.vector(), b)

            # Velocity correction
            b = assemble(L_u_corr)
            for bc in bcu: bc.apply(A_u_corr, b)

            solve(A_u_corr, u1.vector(), b)

            # Rotate functions for next timestep
            u0.assign(u1)
            eta0.assign(eta1)

            # Increase the adjoint timestep
            adj_inc_timestep(time=float(t), finished=self._finished(t,
                finish_time))

            yield({"time": t,
                   "u": u0,
                   "eta": eta0,
                   "is_final": self._finished(t, finish_time)})

        log(INFO, "End of time loop.")
