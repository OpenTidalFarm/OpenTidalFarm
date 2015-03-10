import os.path

from dolfin import *
from dolfin_adjoint import *

from .. import finite_elements
from ..problems import SWProblem
from ..problems import SteadySWProblem
from ..helpers import FrozenClass
from solver import Solver
from les import LES


class IPCSSWSolverParameters(FrozenClass):
    """ A set of parameters for a :class:`IPCSSWSolver`.

    Performance parameters:

    :ivar dolfin_solver: The dictionary with parameters for the dolfin
        Newton solver. A list of valid entries can be printed with:

        .. code-block:: python

            info(NonlinearVariationalSolver.default_parameters(), True)

        By default, the MUMPS direct solver is used for the linear system. If
        not availabe, the default solver and preconditioner of FEniCS is used.

    :ivar quadrature_degree: The quadrature degree for the matrix assembly.
        Default: -1 (auto)
    :ivar cpp_flags: A list of cpp compiler options for the code generation.
        Default: ["-O3", "-ffast-math", "-march=native"]

    Large eddy simulation parameters:

    :ivar les_model: De-/Activates the LES model. Default: True
    :ivar les_model_parameters: A dictionary with parameters for the LES model.
        Default: {'smagorinsky_coefficient': 1e-2}

    """

    dolfin_solver = {"newton_solver": {}}

    # Large eddy simulation
    les_model = True
    les_parameters = {'smagorinsky_coefficient': 1e-2}

    # Performance settings
    quadrature_degree = -1
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
    scheme that follows the idea of Goda [1]_ and Simo [2]_.  This scheme
    preserves the exact same stability properties as Navier-Stokes and hence
    does not introduce additional dissipation in the flow (FIXME: needs
    verification).

    The idea is to replace the unknown free-surface with an approximation. This
    is chosen as the free-surface solution from the previous solution.

    The time discretization is done using a :math:`\theta`-scheme, the
    convection, friction and divergence are handled semi-implicitly. Thus, we have a discretized version
    of the shallow water equations as

    .. math:: \frac{1}{\Delta t}\left( u^{n+1}-u^{n}
        \right)-\nabla\cdot\nu\nabla u^{n+\theta}+u^*\cdot\nabla u^{n+\theta}+g\nabla
        \eta^{n+\theta} + \frac{c_b + c_t}{H^n} \| u^n\| u^{n+\theta} &= f_u^{n+\theta}, \\
        \frac{1}{\Delta t}\left( \eta^{n+1}-\eta^{n} \right) + \nabla \cdot
        \left( H^{n} u^{n+\theta} \right) &= 0,

    where :math:`\square^{n+\theta} = \theta{\square}^{n+1}+(1-\theta)\square^n, \theta \in [0,
    1]` and :math:`u^* = \frac{3}{2}u^n - \frac{1}{2}u^{n-1}`.

    This convection term is unconditionally stable, and with :math:`\theta=0.5`,
    this equation is second order in time and space [2]_ (FIXME: Needs
    verification).

    For the operator splitting, we use the free-surface solution from the
    previous timestep as an estimation, giving an equation for a tentative
    velocity, :math:`\tilde{u}^{n+1}`:

    .. math:: \frac{1}{\Delta t}\left( \tilde{u}^{n+1}-u^{n}
        \right) - \nabla\cdot\nu\nabla \tilde{u}^{n+\theta} + u^*\cdot\nabla \tilde u^{n+\theta}+g\nabla
        \eta^{n}  + \frac{c_b + c_t}{H^n} \| u^n\| \tilde u^{n+\theta} = f_u^{n+\theta}.

    This tenative velocity does not satisfy the divergence equation, and thus we
    define a velocity correction :math:`u^c=u^{n+1}-\tilde{u}^{n+1}`.
    Substracting the second equation from the first, we see that

    .. math::
        \frac{1}{\Delta t}u^c - \theta \nabla\cdot\nu\nabla u^c + \theta
        u^*\cdot\nabla u^{c} + g\theta \nabla\left( \eta^{n+1} - \eta^n\right) +
        \theta \frac{c_b + c_t}{H^n} \| u^n\| u^{c} &= 0, \\
                \frac{1}{\Delta t}\left( \eta^{n+1}-\eta^{n} \right) + \theta \nabla
                \cdot \left( H^{n} u^c \right) &= -\nabla \cdot \left( H^{n}
                \tilde{u}^{n+\theta} \right).

    The operator splitting is a first order approximation, :math:`O(\Delta t)`,
    so we can, without reducing the order of the approximation simplify the
    above to

    .. math::
        \frac{1}{\Delta t}u^c + g\theta \nabla\left( \eta^{n+1} - \eta^n\right) &= 0, \\
        \frac{1}{\Delta t}\left( \eta^{n+1}-\eta^{n} \right) + \theta \nabla \cdot
        \left( H^{n} u^c \right) &= -\nabla \cdot \left( H^{n}
        \tilde{u}^{n+\theta} \right),

    which is reducible to the problem:

    .. math::
        \eta^{n+1}-\eta^{n} - g \Delta t^2 \theta^2 \nabla \cdot
        \left( H^{n+1}  \nabla\left( \eta^{n+1} - \eta^n\right) \right) =
        -\Delta t \nabla \cdot \left( H^{n}
        \tilde{u}^{n+\theta} \right).

    The corrected velocity is then easily calculated from

    .. math::
        u^{n+1} = \tilde{u}^{n+1} - \Delta tg\theta\nabla\left(\eta^{n+1}-\eta^n\right)

    The scheme can be summarized in the following steps:
        #. Replace the pressure with a known approximation and solve for a tenative velocity :math:`\tilde u^{n+1}`.

        #. Solve a free-surface correction equation for the free-surface, :math:`\eta^{n+1}`

        #. Use the corrected pressure to find the velocity correction and calculate :math:`u^{n+1}`

        #. Update t, and repeat.

    Remarks:

      - This solver only works with transient problems, that is with
        :class:`opentidalfarm.problems.sw.SWProblem`.

      - This solver supports large eddy simulation (LES). The LES model is
        implemented via the :class:`opentidalfarm.solvers.les.LES` class.

    .. [1] Goda, Katuhiko. *A multistep technique with implicit difference
        schemes for calculating two-or three-dimensional cavity flows.* Journal of
        Computational Physics 30.1 (1979): 76-95.

    .. [2] Simo, J. C., and F. Armero. *Unconditional stability and long-term
        behavior of transient algorithms for the incompressible Navier-Stokes and
        Euler equations.* Computer Methods in Applied Mechanics and Engineering
        111.1 (1994): 111-154.
    """

    def __init__(self, problem, parameters):

        if not isinstance(problem, (SWProblem, SteadySWProblem)):
            raise TypeError, "problem must be of type Problem"

        if not isinstance(parameters, IPCSSWSolverParameters):
            raise TypeError, "parameters must be of type \
IPCSSWSolverParameters."

        self.problem = problem
        self.parameters = parameters
        self.optimisation_iteration = 0

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

    def _generate_strong_bcs(self, dgu):

        if dgu:
            bcu_method = "geometric"
        else:
            bcu_method = "topological"

        bcs = self.problem.parameters.bcs
        facet_ids = self.problem.parameters.domain.facet_ids

        # Generate velocity boundary conditions
        bcs_u = []
        for _, expr, facet_id, _ in bcs.filter("u", "strong_dirichlet"):
            bc = DirichletBC(self.V, expr, facet_ids, facet_id, method=bcu_method)
            bcs_u.append(bc)

        # Generate free-surface boundary conditions
        bcs_eta = []
        for _, expr, facet_id, _ in bcs.filter("eta", "strong_dirichlet"):
            bc = DirichletBC(self.Q, expr, facet_ids, facet_id)
            bcs_eta.append(bc)

        return bcs_u, bcs_eta

    def solve(self, annotate=True):
        ''' Solve the shallow water equations '''

        ############################### Setting up the equations ###########################

        # Initialise solver settings
        if not type(self.problem) == SWProblem:
            raise TypeError("Do not know how to solve problem of type %s." %
                type(self.problem))

        # Get parameters
        problem_params = self.problem.parameters
        solver_params = self.parameters
        farm = problem_params.tidal_farm
        if farm:
            turbine_friction = farm.friction_function

        # Performance settings
        parameters['form_compiler']['quadrature_degree'] = \
            solver_params.quadrature_degree
        parameters['form_compiler']['cpp_optimize_flags'] = \
            " ".join(solver_params.cpp_flags)
        parameters['form_compiler']['cpp_optimize'] = True
        parameters['form_compiler']['optimize'] = True

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
        nu = problem_params.viscosity
        include_advection = problem_params.include_advection
        include_viscosity = problem_params.include_viscosity
        linear_divergence = problem_params.linear_divergence
        f_u = problem_params.f_u
        include_les = solver_params.les_model

        # Get boundary conditions
        bcs = problem_params.bcs

        # Get function spaces
        V, Q = self.V, self.Q
        dgu = "Discontinuous" in str(V)

        # Test and trial functions
        v = TestFunction(V)
        u = TrialFunction(V)
        q = TestFunction(Q)
        eta = TrialFunction(Q)

        # Functions
        u00 = Function(V)
        u0 = Function(V, name="u0")
        ut = Function(V) # Tentative velocity
        u1 = Function(V, name="u")
        eta0 = Function(Q, name="eta0")
        eta1 = Function(Q, name="eta")

        # Large eddy model
        if include_les:
            les_V = FunctionSpace(problem_params.domain.mesh, "CG", 1)
            les = LES(les_V, u0,
                    solver_params.les_parameters['smagorinsky_coefficient'])
            eddy_viscosity = les.eddy_viscosity
            nu += eddy_viscosity
        else:
            eddy_viscosity = None

        # Define the water depth
        if linear_divergence:
            H = h
        else:
            H = eta0 + h

        if f_u is None:
            f_u = Constant((0, 0))

        # Bottom friction
        friction = problem_params.friction

        if farm:
            friction += Function(turbine_friction, name="turbine_friction",
                    annotate=annotate)

        # Load initial conditions
        # Projection is necessary to obtain 2nd order convergence
        # FIXME: The problem should specify the ic for each component separately.
        u_ic = project(problem_params.initial_condition_u, self.V)
        u0.assign(u_ic, annotate=False)
        u00.assign(u_ic, annotate=False)
        eta_ic = project(problem_params.initial_condition_eta, self.Q)
        eta0.assign(eta_ic, annotate=False)
        eta1.assign(eta_ic, annotate=False)

        # Tentative velocity step
        u_mean = theta * u + (1. - theta) * u0
        u_bash = 3./2 * u0 - 1./2 * u00
        u_diff = u - u0
        norm_u0 = inner(u0, u0)**0.5
        F_u_tent = ((1/dt) * inner(v, u_diff) * dx()
                    + inner(v, grad(u_bash)*u_mean) * dx()
                    + g * inner(v, grad(eta0)) * dx()
                    + friction / H * norm_u0 * inner(u_mean, v) * dx
                    - inner(v, f_u) * dx())
        # Viscosity term
        if dgu:
            # Taken from http://maths.dur.ac.uk/~dma0mpj/summer_school/IPHO.pdf
            sigma = 1. # Penalty parameter.
            # Set tau=-1 for SIPG, tau=0 for IIPG, and tau=1 for NIPG
            tau = 0.
            edgelen = FacetArea(self.mesh)('+')  # Facetarea is continuous, so
            # we can select either side
            alpha = sigma/edgelen

            F_u_tent += nu * inner(grad(v), grad(u_mean)) * dx()
            for d in range(2):
                F_u_tent += - nu * inner(avg(grad(u_mean[d])), jump(v[d], n))*dS
                F_u_tent += - nu * tau * inner(avg(grad(v[d])), jump(u[d], n))*dS
                F_u_tent += alpha * nu * inner(jump(u[d], n), jump(v[d], n))*dS

        else:
            F_u_tent += nu * inner(grad(v), grad(u_mean)) * dx()

        a_u_tent = lhs(F_u_tent)
        L_u_tent = rhs(F_u_tent)

        # Pressure correction
        eta_diff = eta - eta0
        ut_mean = theta * ut + (1. - theta) * u0
        F_p_corr = (q*eta_diff + g * dt**2 * theta**2 * H * inner(grad(q),
            grad(eta_diff)))*dx() + dt*q*div(H*ut_mean)*dx()
        a_p_corr = lhs(F_p_corr)
        L_p_corr = rhs(F_p_corr)

        # Velocity correction
        eta_diff = eta1 - eta0
        a_u_corr = inner(v, u)*dx()
        L_u_corr = inner(v, ut)*dx() - dt*g*theta*inner(v, grad(eta_diff))*dx()

        bcu, bceta = self._generate_strong_bcs(dgu)

        # Assemble matrices
        A_u_corr = assemble(a_u_corr)
        for bc in bcu: bc.apply(A_u_corr)
        a_u_corr_solver = LUSolver(A_u_corr)
        a_u_corr_solver.parameters["reuse_factorization"] = True

        if linear_divergence:
            A_p_corr = assemble(a_p_corr)
            for bc in bceta: bc.apply(A_p_corr)
            a_p_corr_solver = LUSolver(A_p_corr)
            a_p_corr_solver.parameters["reuse_factorization"] = True

        yield({"time": t,
               "u": u0,
               "eta": eta0,
               "eddy_viscosity": eddy_viscosity,
               "is_final": self._finished(t, finish_time)})

        log(INFO, "Start of time loop")
        adjointer.time.start(t)
        timestep = 0

        # De/activate annotation
        annotate_orig = parameters["adjoint"]["stop_annotating"]
        parameters["adjoint"]["stop_annotating"] = not annotate

        while not self._finished(t, finish_time):
            # Update timestep
            timestep += 1
            t = Constant(t + dt)

            # Update bc's
            t_theta = Constant(t - (1.0 - theta) * dt)
            bcs.update_time(t, only_type=["strong_dirichlet"])
            bcs.update_time(t_theta, exclude_type=["strong_dirichlet"])

            # Update source term
            if f_u is not None:
                f_u.t = Constant(t_theta)

            if include_les:
                log(PROGRESS, "Compute eddy viscosity.")
                les.solve()

            # Compute tentative velocity step
            log(PROGRESS, "Solve for tentative velocity.")
            A_u_tent = assemble(a_u_tent)
            b = assemble(L_u_tent)
            for bc in bcu: bc.apply(A_u_tent, b)

            solve(A_u_tent, ut.vector(), b)

            # Pressure correction
            log(PROGRESS, "Solve for pressure correction.")
            b = assemble(L_p_corr)
            for bc in bceta: bc.apply(b)

            if linear_divergence:
                a_p_corr_solver.solve(eta1.vector(), b)
            else:
                A_p_corr = assemble(a_p_corr)
                for bc in bceta: bc.apply(A_p_corr)
                solve(A_p_corr, eta1.vector(), b)

            # Velocity correction
            log(PROGRESS, "Solve for velocity update.")
            b = assemble(L_u_corr)
            for bc in bcu: bc.apply(b)

            a_u_corr_solver.solve(u1.vector(), b)

            # Rotate functions for next timestep
            u00.assign(u0)
            u0.assign(u1)
            eta0.assign(eta1)

            # Increase the adjoint timestep
            adj_inc_timestep(time=float(t), finished=self._finished(t,
                finish_time))

            yield({"time": t,
                   "u": u0,
                   "eta": eta0,
                   "eddy_viscosity": eddy_viscosity,
                   "is_final": self._finished(t, finish_time)})

        # Reset annotation flag
        parameters["adjoint"]["stop_annotating"] = annotate_orig

        log(INFO, "End of time loop.")
