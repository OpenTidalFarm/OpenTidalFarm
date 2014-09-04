import os.path

from dolfin import *
from dolfin_adjoint import *

from solver import Solver
from ..problems import SWProblem
from ..problems import SteadySWProblem
from ..problems import MultiSteadySWProblem
from ..helpers import StateWriter, norm_approx, smooth_uflmin, FrozenClass


class CoupledSWSolverParameters(FrozenClass):
    """ A set of parameters for a :class:`SteadyCoupledSWSolver`. 

    Following parameters are available:

    :ivar dolfin_solver: The dictionary with parameters for the dolfin
        Newton solver. A list of valid entries look at:

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
        Default: -1 (automatic)
    :ivar cpp_flags: A list of cpp compiler options for the code generation.
        Default: ["-O3", "-ffast-math", "-march=native"]

    """

    dolfin_solver = {"newton_solver": {}}
    dump_period = 1
    print_individual_turbine_power = False

    # Performance settings
    cache_forward_state = True
    quadrature_degree = -1
    cpp_flags = ["-O3", "-ffast-math", "-march=native"]

    def __init__(self):

        linear_solver = 'mumps' if ('mumps' in map(lambda x: x[0], linear_solver_methods())) else 'default'
        preconditioner = 'default'

        self.dolfin_solver["newton_solver"]["linear_solver"] = linear_solver
        self.dolfin_solver["newton_solver"]["preconditioner"] = preconditioner
        self.dolfin_solver["newton_solver"]["maximum_iterations"] = 20


class CoupledSWSolver(Solver):

    def __init__(self, problem, parameters, config=None):

        if not isinstance(problem, (SWProblem,
            SteadySWProblem)):
            raise TypeError, "problem must be of type Problem"

        if not isinstance(parameters, CoupledSWSolverParameters):
            raise TypeError, "parameters must be of type \
CoupledSWSolverParameters."

        self.problem = problem
        self.parameters = parameters
        self.config = config

        # If cache_for_nonlinear_initial_guess is true, then we store all
        # intermediate state variables in this dictionary to be used for the
        # next solve
        self.state_cache = {}

        self.current_state = None

        self.mesh = problem.parameters.domain.mesh
        V, H = self.problem.parameters.finite_element(self.mesh)
        self.function_space = MixedFunctionSpace([V, H])


    @staticmethod
    def default_parameters():
        return CoupledSWSolverParameters()

    def _finished(self, current_time, finish_time):
        return float(current_time - finish_time) >= - 1e3*DOLFIN_EPS

    def solve(self, turbine_field=None, annotate=True, u_source=None):
        ''' Solve the shallow water equations '''

        ############################### Setting up the equations ###########################

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

        # Get domain measures
        ds = problem_params.domain.ds

        # Initialise solver settings
        if type(self.problem) == SWProblem:
            log(INFO, "Solve a transient shallow water problem")

            theta = Constant(problem_params.theta)
            dt = Constant(problem_params.dt)
            finish_time = Constant(problem_params.finish_time)

            t = Constant(problem_params.start_time)

            include_time_term = True

        elif type(self.problem) == MultiSteadySWProblem:
            log(INFO, "Solve a multi steady-state shallow water problem")

            theta = Constant(1)
            dt = Constant(problem_params.dt)
            finish_time = Constant(problem_params.finish_time)

            # The multi steady-state case solves the steady-state equation also
            # for the start time
            t = Constant(problem_params.start_time - dt)

            include_time_term = False

        elif type(self.problem) == SteadySWProblem:
            log(INFO, "Solve a steady-state shallow water problem")

            theta = Constant(1.)
            dt = Constant(1.)
            finish_time = Constant(0.5)

            t = Constant(0.)

            include_time_term = False
        
        else:
            raise TypeError("Do not know how to solve problem of type %s." % 
                type(problem))

        g = problem_params.g
        depth = problem_params.depth
        include_advection = problem_params.include_advection
        include_viscosity = problem_params.include_viscosity
        viscosity = problem_params.viscosity
        bcs = problem_params.bcs
        linear_divergence = problem_params.linear_divergence
        cache_forward_state = solver_params.cache_forward_state
        
        # Define test functions
        v, q = TestFunctions(self.function_space)

        # Define functions
        state = Function(self.function_space, name="Current_state")
        self.current_state = state
        state_new = Function(self.function_space, name="New_state")

        # Load initial condition (or initial guess for stady problems)
        # Projection is necessary to obtain 2nd order convergence
        ic = project(problem_params.initial_condition, self.function_space)
        state.assign(ic, annotate=False)    

        # Split mixed functions
        u, h = split(state_new)

        u0, h0 = split(state)

        # Define the water depth
        if linear_divergence:
            H = depth
        else:
            H = h + depth

        # Create initial conditions and interpolate
        state_new.assign(state, annotate=annotate)

        # u_(n+theta) and h_(n+theta)
        u_mid = (1.0 - theta) * u0 + theta * u
        h_mid = (1.0 - theta) * h0 + theta * h

        # The normal direction
        n = FacetNormal(self.mesh)

        # Mass matrix contributions
        M = inner(v, u) * dx
        M += inner(q, h) * dx
        M0 = inner(v, u0) * dx
        M0 += inner(q, h0) * dx

        # Divergence term.
        Ct_mid = -H * inner(u_mid, grad(q)) * dx
        #+inner(avg(u_mid),jump(q,n))*dS # This term is only needed for dg 
                                         # element pairs which is not supported

        # The surface integral contribution from the divergence term
        bc_contr = -H * dot(u_mid, n) * q * ds

        for function_name, u_expr, facet_id, bctype in problem_params.bcs:
            if (function_name is not "u" or 
                bctype not in ["flather", "weak_dirichlet"]):
                continue

            if bctype == 'weak_dirichlet':
                # Subtract the divergence integral again
                bc_contr -= -H * dot(u_mid, n) * q * ds(facet_id)
                bc_contr -= H * dot(u_expr, n) * q * ds(facet_id)

            elif bctype == 'flather':
                # The Flather boundary condition requires two facet_ids.
                assert len(facet_id) == 2

                # Subtract the divergence integrals again
                bc_contr -= -H * dot(u_mid, n) * q * ds(facet_id[0])
                bc_contr -= -H * dot(u_mid, n) * q * ds(facet_id[1])

                # The Flather boundary condition on the left hand side
                bc_contr -= H * dot(u_expr, n) * q * ds(facet_id[0])
                Ct_mid += sqrt(g * H) * inner(h_mid, q) * ds(facet_id[0])

                # The contributions of the Flather boundary condition on the right hand side
                Ct_mid += sqrt(g * H) * inner(h_mid, q) * ds(facet_id[1])


        # Pressure gradient term
        weak_eta_bcs = [bc for bc in problem_params.bcs if bc[0] is "eta" 
                        and bc[-1] is "weak_dirichlet"]

        if len(weak_eta_bcs) == 0:
            # assume we don't want to integrate the pressure gradient by parts
            C_mid = g * inner(v, grad(h_mid)) * dx

        else:
            C_mid = -g * inner(div(v), h_mid) * dx
            C_mid += g * inner(dot(v, n), h_mid) * ds

            for function_name, eta_expr, facet_id, bctype in weak_eta_bcs:
                # Remove the original boundary integral of that facet
                C_mid -= g * inner(dot(v, n), h_mid) * ds(facet_id)

                # Apply the eta boundary conditions weakly on boundary IDs 1 and 2
                C_mid +=  g * inner(dot(v, n), eta_expr) * ds(facet_id)

                #+inner(avg(v),jump(h_mid,n))*dS # This term is only needed 
                                                 # for dg element pairs

        # Bottom friction
        friction = problem_params.friction

        if not turbine_field:
            tf = Constant(0)
        elif type(turbine_field) == list:
            tf = Function(turbine_field[0], name="turbine_friction", annotate=annotate)
        else:
            tf = Function(turbine_field, name="turbine_friction", annotate=annotate)

        # Friction term
        # FIXME: FEniCS fails on assembling the below form for u_mid = 0, even
        # though it is differentiable. Even this potential fix does not help:
        #norm_u_mid = conditional(inner(u_mid, u_mid)**0.5 < DOLFIN_EPS, Constant(0),
        #        inner(u_mid, u_mid)**0.5)
        norm_u_mid = inner(u_mid, u_mid)**0.5
        R_mid = friction / H * norm_u_mid * inner(u_mid, v) * dx

        if turbine_field:
            R_mid += tf / H * dot(u_mid, u_mid) ** 0.5 * inner(u_mid, v) * self.config.site_dx(1)

        # Advection term
        if include_advection:
            Ad_mid = inner(dot(grad(u_mid), u_mid), v) * dx

        if include_viscosity:
            # Check that we are not using a DG velocity function space, as the facet integrals are not implemented.
            if "Discontinuous" in str(self.function_space.split()[0]):
                raise NotImplementedError("The viscosity term for \
                    discontinuous elements is not supported.")
            D_mid = viscosity * inner(grad(u_mid), grad(v)) * dx

        # Create the final form
        G_mid = C_mid + Ct_mid + R_mid

        # Add the advection term
        if include_advection:
            G_mid += Ad_mid

        # Add the viscosity term
        if include_viscosity:
            G_mid += D_mid

        # Add the source term
        if u_source:
            G_mid -= inner(u_source, v) * dx
        F = dt * G_mid - dt * bc_contr

        # Add the time term
        if include_time_term:
            F += M - M0

        ############################### Perform the simulation ###########################

        if solver_params.dump_period > 0:

            writer = StateWriter(self.config, optimisation_iteration=self.config.optimisation_iteration)
            if type(self.problem) == SWProblem:
                log(INFO, "Writing state to disk...")
                writer.write(state)

        step = 0

        yield({"time": t, 
               "u": u0,
               "eta": h0,
               "tf": tf,
               "state": state,
               "is_final": self._finished(t, finish_time)})

        log(INFO, "Start of time loop")
        adjointer.time.start(t)
        timestep = 0
        while not self._finished(t, finish_time):
            timestep += 1
            t = Constant(t + dt)

            # Update bc's
            t_theta = Constant(t - (1.0 - theta) * dt)
            bcs.update_time(t, only_type=["strong_dirichlet"])
            bcs.update_time(t_theta, exclude_type=["strong_dirichlet"])

            # Update source term
            if u_source is not None:
                u_source.t = Constant(t_theta)
            step += 1

            # Set the initial guess for the solve
            if cache_forward_state and self.state_cache.has_key(float(t)):
                log(INFO, "Read initial guess from cache for t=%f." % t)
                # Load initial guess for solver from cache
                state_new.assign(self.state_cache[float(t)], annotate=False)

            elif not include_time_term:
                log(INFO, "Set the initial guess for the nonlinear solver to the initial condition.")
                # Reset the initial guess after each timestep
                ic = problem_params.initial_condition
                state_new.assign(ic, annotate=False)

            # Solve non-linear system with a Newton solver
            if self.problem._is_transient:
                log(INFO, "Solve shallow water equations at time %s" % float(t))
            else:
                log(INFO, "Solve shallow water equations.")

            strong_bcs = []
            for function_name, expr, facet_id, bctype in problem_params.bcs:
                if bctype is not "strong_dirichlet":
                    continue

                if function_name is "u":
                    subidx = 0
                elif function_name is "eta":
                    subidx = 1

                bc = DirichletBC(self.function_space.sub(subidx), expr,
                                 problem_params.domain.facet_ids, facet_id)
                strong_bcs.append(bc)

            solve(F == 0, state_new, bcs=strong_bcs,
                  solver_parameters=solver_params.dolfin_solver, 
                  annotate=annotate, 
                  J=derivative(F, state_new))

            # After the timestep solve, update state
            state.assign(state_new)

            if cache_forward_state:
                # Save state for initial guess cache
                log(INFO, "Cache solution t=%f as next initial guess." % t)
                if not self.state_cache.has_key(float(t)):
                    self.state_cache[float(t)] = Function(self.function_space)
                self.state_cache[float(t)].assign(state_new, annotate=False)

            # Set the control function for the upcoming timestep.
            if turbine_field:
                if type(turbine_field) == list:
                    tf.assign(turbine_field[timestep])
                else:
                    tf.assign(turbine_field)

            if (solver_params.dump_period > 0 and 
                step % solver_params.dump_period == 0):
                log(INFO, "Write state to disk...")
                writer.write(state)


            # Increase the adjoint timestep
            adj_inc_timestep(time=float(t), finished=self._finished(t,
                finish_time))

            yield({"time": t, 
                   "u": u0,
                   "eta": h0,
                   "tf": tf,
                   "state": state,
                   "is_final": self._finished(t, finish_time)})

        log(INFO, "End of time loop.")
