import os.path

from dolfin import *
from dolfin_adjoint import *

from solver import Solver
from ..problems import ShallowWaterProblem, SteadyShallowWaterProblem
from ..helpers import StateWriter, norm_approx, smooth_uflmin, FrozenClass


class ShallowWaterSolverParameters(FrozenClass):
    """ A set of parameters for a :class:`SteadyShallowWaterSolver`. 

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

    """

    dolfin_solver = {"newton_solver": {}}
    dump_period = 1
    print_individual_turbine_power = False

    # Performance settings
    cache_forward_state = True

    def __init__(self):

        linear_solver = 'mumps' if ('mumps' in map(lambda x: x[0], linear_solver_methods())) else 'default'
        preconditioner = 'default'

        self.dolfin_solver["newton_solver"]["linear_solver"] = linear_solver
        self.dolfin_solver["newton_solver"]["preconditioner"] = preconditioner
        self.dolfin_solver["newton_solver"]["maximum_iterations"] = 20


class ShallowWaterSolver(Solver):

    def __init__(self, problem, parameters, config=None):

        if not isinstance(problem, (ShallowWaterProblem,
            SteadyShallowWaterProblem)):
            raise TypeError, "problem must be of type Problem"

        if not isinstance(parameters, ShallowWaterSolverParameters):
            raise TypeError, "parameters must be of type \
ShallowWaterSolverParameters."

        self.problem = problem
        self.parameters = parameters
        self.config = config

        # If cache_for_nonlinear_initial_guess is true, then we store all
        # intermediate state variables in this dictionary to be used for the
        # next solve
        self.state_cache = {}

        self.current_state = None

    @staticmethod
    def default_parameters():
        return ShallowWaterSolverParameters()

    def solve(self, turbine_field=None, functional=None, annotate=True, u_source=None):
        ''' Solve the shallow water equations '''

        ############################### Setting up the equations ###########################

        # Define variables for all used parameters
        problem_params = self.problem.parameters

        mesh = problem_params.domain.mesh
        ds = problem_params.domain.ds

        solver_params = self.parameters

        if self.problem._is_transient:

            theta = Constant(problem_params.theta)
            dt = Constant(problem_params.dt)
            finish_time = Constant(problem_params.finish_time)
            start_time = Constant(problem_params.start_time)

            # Reset the time
            problem_params.current_time = Constant(problem_params.start_time)
            t = Constant(problem_params.current_time)

            functional_final_time_only = problem_params.functional_final_time_only
            functional_quadrature_degree = problem_params.functional_quadrature_degree

            include_time_term = problem_params.include_time_term

        else:
            theta = Constant(1.)
            dt = Constant(1.)
            start_time = Constant(0.)
            finish_time = Constant(0.5)
            t = start_time

            functional_final_time_only = True
            functional_quadrature_degree = 1

            include_time_term = False

        g = problem_params.g
        depth = problem_params.depth
        include_advection = problem_params.include_advection
        include_viscosity = problem_params.include_viscosity
        viscosity = problem_params.viscosity
        bcs = problem_params.bcs
        linear_divergence = problem_params.linear_divergence
        cache_forward_state = solver_params.cache_forward_state
        
        if not 0 <= functional_quadrature_degree <= 1:
            raise ValueError("functional_quadrature_degree must be 0 or 1.")

        V, H = problem_params.finite_element(mesh)
        function_space = MixedFunctionSpace([V, H])

        # Define test functions
        v, q = TestFunctions(function_space)


        # Define functions
        state = Function(function_space, name="Current_state")  # solution of the next timestep
        self.current_state = state
        state_new = Function(function_space, name="New_state")  # solution of the next timestep

        # Load initial condition (or initial guess for stady problems)
        ic = project(problem_params.initial_condition, function_space)
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
        n = FacetNormal(function_space.mesh())

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

        if turbine_field:
            if type(turbine_field) == list:
                tf = Function(turbine_field[0], name="turbine_friction", annotate=annotate)
            else:
                tf = Function(turbine_field, name="turbine_friction", annotate=annotate)

        # Friction term
        R_mid = friction / H * dot(u_mid, u_mid) ** 0.5 * inner(u_mid, v) * dx

        if turbine_field:
            R_mid += tf / H * dot(u_mid, u_mid) ** 0.5 * inner(u_mid, v) * self.config.site_dx(1)

        # Advection term
        if include_advection:
            Ad_mid = inner(dot(grad(u_mid), u_mid), v) * dx

        if include_viscosity:
            # Check that we are not using a DG velocity function space, as the facet integrals are not implemented.
            if "Discontinuous" in str(function_space.split()[0]):
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
            try:
                statewriter_cb = self.config.statewriter_callback
            except AttributeError:
                statewriter_cb = None 

            writer = StateWriter(self.config, optimisation_iteration=self.config.optimisation_iteration, callback=statewriter_cb)
            if self.problem._is_transient and include_time_term:
                log(INFO, "Writing state to disk...")
                writer.write(state)

        step = 0

        if functional is not None:
            if not self.problem._is_transient or functional_final_time_only:
                j = 0.
                if solver_params.print_individual_turbine_power:
                    j_individual = [0] * len(params["turbine_pos"])
                    force_individual = [0] * len(params["turbine_pos"])

            else:
                if functional_quadrature_degree == 0:
                    quad = 0.0
                else:
                    quad = 0.5
                j = dt * quad * assemble(functional.Jt(state, tf))
                if solver_params.print_individual_turbine_power:
                    j_individual = []
                    force_individual = []
                    for i in range(len(params["turbine_pos"])):
                        j_individual.append(dt * quad * assemble(functional.Jt_individual(state, i)))
                        force_individual.append(dt * quad * assemble(functional.force_individual(state, i)))

        log(INFO, "Start of time loop")
        adjointer.time.start(t)
        timestep = 0
        while (float(t) < float(finish_time)):
            timestep += 1
            t = Constant(t + dt)
            if self.problem._is_transient:
                problem_params.current_time = t

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
                log(INFO, "Solve shallow water equations at time %s (Newton iteration) ..." % float(problem_params.current_time))
            else:
                log(INFO, "Solve shallow water equations (Newton iteration) \
...")

            strong_bcs = []
            for function_name, expr, facet_id, bctype in problem_params.bcs:
                if bctype is not "strong_dirichlet":
                    continue

                if function_name is "u":
                    subidx = 0
                elif function_name is "eta":
                    subidx = 1

                bc = DirichletBC(function_space.sub(subidx), expr,
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
                    self.state_cache[float(t)] = Function(function_space)
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

            if functional is not None:
                if not (functional_final_time_only and 
                        float(t) < float(finish_time)):
                    if not self.problem._is_transient or functional_final_time_only or functional_quadrature_degree == 0:
                        quad = 1.0
                    elif float(t) >= float(finish_time):
                        quad = 0.5 * dt
                    else:
                        quad = 1.0 * dt

                    j += quad * assemble(functional.Jt(state, tf))
                    if solver_params.print_individual_turbine_power:
                        log(INFO, "Computing individual turbine power extraction contribution...")
                        individual_contribution_list = ['x_pos', 'y_pos', 'turbine_power', 'total_force_on_turbine', 'turbine_friction']
                        fr_individual = range(len(params["turbine_pos"]))
                        for i in range(len(params["turbine_pos"])):
                            j_individual[i] += dt * quad * assemble(functional.Jt_individual(state, i))
                            force_individual[i] += dt * quad * assemble(functional.force_individual(state, i))

                            if len(params["turbine_friction"]) > 0:
                                fr_individual[i] = params["turbine_friction"][i]
                            else:
                                fr_individual = [params["turbine_friction"]] * len(params["turbine_pos"])

                            individual_contribution_list.append((params["turbine_pos"][i])[0])
                            individual_contribution_list.append((params["turbine_pos"][i])[1])
                            individual_contribution_list.append(j_individual[i])
                            individual_contribution_list.append(force_individual[i])
                            individual_contribution_list.append(fr_individual[i])

                            log(INFO, "Contribution of turbine number %d at co-ordinates:" % (i + 1), params["turbine_pos"][i], ' is: ', j_individual[i] * 0.001, 'kW', 'with friction of', fr_individual[i])

            # Increase the adjoint timestep
            adj_inc_timestep(time=float(t), finished=(not float(t) < float(finish_time)))
        log(INFO, "End of time loop.")

        # Write the turbine positions, power extraction and friction to a .csv file named turbine_info.csv
        if solver_params.print_individual_turbine_power:
            f = self.config.params['base_path'] + os.path.sep + "iter_" + \
                str(self.config.optimisation_iteration) + '/'
            # Save the very first result in a different file
            if self.config.optimisation_iteration == 0 and not os.path.isfile(f):
                f += 'initial_turbine_info.csv'
            else:
                f += 'turbine_info.csv'

            output_turbines = open(f, 'w')
            for i in range(0, len(individual_contribution_list), 5):
                print >> output_turbines, '%s, %s, %s, %s, %s' % (individual_contribution_list[i], individual_contribution_list[i + 1], individual_contribution_list[i + 2], individual_contribution_list[i + 3], individual_contribution_list[i + 4])
            print 'Total of individual turbines is', sum(j_individual)

        if functional is not None:
            return j
        else:
            return state_new
