import sys
import os.path

from dolfin import *
from dolfin_adjoint import *

from solver import Solver
from ..helpers import StateWriter, norm_approx, smooth_uflmin



def default_solver_parameters():
    ''' Create a dictionary with the default solver parameters '''
    linear_solver = 'mumps' if ('mumps' in map(lambda x: x[0], linear_solver_methods())) else 'default'
    preconditioner = 'default'

    solver_parameters = {"newton_solver": {}}

    solver_parameters["newton_solver"]["linear_solver"] = linear_solver
    solver_parameters["newton_solver"]["preconditioner"] = preconditioner
    solver_parameters["newton_solver"]["maximum_iterations"] = 20

    return solver_parameters


class ShallowWaterSolver(Solver):

    def __init__(self, config):
        self.config = config

        # If cache_for_nonlinear_initial_guess is true, then we store all
        # intermediate state variables in this dictionary to be used for the
        # next solve
        self.state_cache = {}

    def solve(self, state, turbine_field=None, functional=None, annotate=True, u_source=None):
        ''' Solve the shallow water equations '''

        config = self.config

        ############################### Setting up the equations ###########################

        # Define variables for all used parameters
        ds = config.domain.ds
        params = config.params

        # To begin with, check if the provided parameters are valid
        params.check()

        theta = params["theta"]
        dt = float(params["dt"])
        g = params["g"]
        depth = params["depth"]
        # Reset the time
        params["current_time"] = params["start_time"]
        t = float(params["current_time"])
        include_advection = params["include_advection"]
        include_viscosity = params["include_viscosity"]
        include_time_term = params["include_time_term"]
        viscosity = params["viscosity"]
        solver_parameters = params["solver_parameters"]
        if solver_parameters is None:
            solver_parameters = default_solver_parameters()
        bctype = params["bctype"]
        strong_bc = params["strong_bc"]
        free_slip_on_sides = params["free_slip_on_sides"]
        steady_state = params["steady_state"]
        linear_divergence = params["linear_divergence"]
        functional_final_time_only = params["functional_final_time_only"]
        functional_quadrature_degree = params["functional_quadrature_degree"]
        cache_forward_state = params["cache_forward_state"]
        postsolver_callback = params["postsolver_callback"]
        
        if not 0 <= functional_quadrature_degree <= 1:
            raise ValueError("functional_quadrature_degree must be 0 or 1.")

        function_space = config.function_space

        # Take care of the steady state case
        if steady_state:
            dt = 1.
            params["finish_time"] = params["start_time"] + dt / 2
            theta = 1.

        # Define test functions
        v, q = TestFunctions(function_space)

        # Define functions
        state_new = Function(function_space, name="New_state")  # solution of the next timestep

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

        # Mass matrix

        M = inner(v, u) * dx
        M += inner(q, h) * dx
        M0 = inner(v, u0) * dx
        M0 += inner(q, h0) * dx

        # Divergence term.
        Ct_mid = -H * inner(u_mid, grad(q)) * dx
        #+inner(avg(u_mid),jump(q,n))*dS # This term is only needed for dg element pairs

        if bctype == 'dirichlet':
            if steady_state:
                raise ValueError("Can not use a time dependent boundary condition for a steady state simulation")
            # The dirichlet boundary condition on the left hand side
            u_expr = config.params["u_weak_dirichlet_bc_expr"]
            bc_contr = - H * dot(u_expr, n) * q * ds(1)

            # The dirichlet boundary condition on the right hand side
            bc_contr -= H * dot(u_expr, n) * q * ds(2)

            # We enforce a no-normal flow on the sides by removing the surface integral.
            # bc_contr -= dot(u_mid, n) * q * ds(3)

        elif bctype == 'flather':
            if steady_state:
                raise ValueError("Can not use a time dependent boundary condition for a steady state simulation")
            # The Flather boundary condition on the left hand side
            u_expr = config.params["flather_bc_expr"]
            bc_contr = - H * dot(u_expr, n) * q * ds(1)
            Ct_mid += sqrt(g * H) * inner(h_mid, q) * ds(1)

            # The contributions of the Flather boundary condition on the right hand side
            Ct_mid += sqrt(g * H) * inner(h_mid, q) * ds(2)

        elif bctype == 'strong_dirichlet':
            # Do not replace anything in the surface integrals as the strong Dirichlet Boundary condition will do that
            bc_contr = -H * dot(u_mid, n) * q * ds(1)
            bc_contr -= H * dot(u_mid, n) * q * ds(2)
            if not free_slip_on_sides:
                bc_contr -= H * dot(u_mid, n) * q * ds(3)

        else:
            log(ERROR, "Unknown boundary condition type: %s" % bctype)
            sys.exit(1)

        # Pressure gradient operator
        eta_expr = config.params["eta_weak_dirichlet_bc_expr"]
        if eta_expr is None: # assume we don't want to integrate the pressure gradient by parts
          C_mid = g * inner(v, grad(h_mid)) * dx
          #+inner(avg(v),jump(h_mid,n))*dS # This term is only needed for dg element pairs
        else:
          # Apply the eta boundary conditions weakly on boundary IDs 1 and 2
          C_mid =  -g * inner(div(v), h_mid) * dx
          C_mid +=  g * inner(dot(v, n), eta_expr) * ds(1)
          C_mid +=  g * inner(dot(v, n), eta_expr) * ds(2)
          C_mid +=  g * inner(dot(v, n), h_mid)    * ds(3)

        # Bottom friction
        friction = params["friction"]

        if turbine_field:
            if type(turbine_field) == list:
                tf = Function(turbine_field[0], name="turbine_friction", annotate=annotate)
            else:
                tf = Function(turbine_field, name="turbine_friction", annotate=annotate)

        # Friction term
        R_mid = friction / H * dot(u_mid, u_mid) ** 0.5 * inner(u_mid, v) * dx

        if turbine_field:
            R_mid += tf / H * dot(u_mid, u_mid) ** 0.5 * inner(u_mid, v) * config.site_dx(1)

        # Advection term
        if include_advection:
            Ad_mid = inner(dot(grad(u_mid), u_mid), v) * dx

        if include_viscosity:
            # Check that we are not using a DG velocity function space, as the facet integrals are not implemented.
            if "Discontinuous" in str(function_space.split()[0]):
                raise NotImplementedError("The viscosity term for discontinuous elements is not implemented yet.")
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
        if include_time_term and not steady_state:
            F += M - M0

        # Do some parameter checking:
        if "dynamic_turbine_friction" in params["controls"]:
            if len(config.params["turbine_friction"]) != (float(params["finish_time"]) - t) / dt + 1:
                log(INFO, "You control the turbine friction dynamically, but your turbine friction parameter is not an array of length 'number of timesteps' (here: %i)." % ((float(params["finish_time"]) - t) / dt + 1))
                import sys
                sys.exit(1)

        ############################### Perform the simulation ###########################

        if params["dump_period"] > 0:
            try:
                statewriter_cb = config.statewriter_callback
            except AttributeError:
                statewriter_cb = None 

            writer = StateWriter(config, optimisation_iteration=config.optimisation_iteration, callback=statewriter_cb)
            if not steady_state and include_time_term:
                log(INFO, "Writing state to disk...")
                writer.write(state)

        step = 0

        if functional is not None:
            if steady_state or functional_final_time_only:
                j = 0.
                if params["print_individual_turbine_power"]:
                    j_individual = [0] * len(params["turbine_pos"])
                    force_individual = [0] * len(params["turbine_pos"])

            else:
                if functional_quadrature_degree == 0:
                    quad = 0.0
                else:
                    quad = 0.5
                j = dt * quad * assemble(functional.Jt(state, tf))
                if params["print_individual_turbine_power"]:
                    j_individual = []
                    force_individual = []
                    for i in range(len(params["turbine_pos"])):
                        j_individual.append(dt * quad * assemble(functional.Jt_individual(state, i)))
                        force_individual.append(dt * quad * assemble(functional.force_individual(state, i)))

        log(INFO, "Start of time loop")
        adjointer.time.start(t)
        timestep = 0
        while (t < float(params["finish_time"])):
            timestep += 1
            t += dt
            params["current_time"] = t

            # Update bc's
            if bctype == "strong_dirichlet":
                strong_bc.update_time(t)
            else:
                u_expr.t = t - (1.0 - theta) * dt

            if eta_expr is not None:
                eta_expr.t = t - (1.0 - theta) * dt

            # Update source term
            if u_source:
                u_source.t = t - (1.0 - theta) * dt
            step += 1

            # Solve non-linear system with a Newton solver
            if cache_forward_state and self.state_cache.has_key(t):
                log(INFO, "Load initial guess from cache for time %f." % t)
                # Load initial guess for solver from cache
                state_new.assign(self.state_cache[t], annotate=False)
            elif not include_time_term:
                log(INFO, "Set the initial guess for the nonlinear solver to the initial condition.")
                # Reset the initial guess after each timestep
                ic = config.params['initial_condition']
                state_new.assign(ic, annotate=False)

            log(INFO, "Solve shallow water equations at time %s (Newton iteration) ..." % float(params["current_time"]))
            if bctype == 'strong_dirichlet':
                F_bcs = strong_bc.bcs
            else:
                F_bcs = []

            solver = config.params['nonlinear_solver']
            if solver is None:
              solve(F == 0, state_new, bcs=F_bcs, solver_parameters=solver_parameters, annotate=annotate, J=derivative(F, state_new))
            else:
              solver(F, state_new, F_bcs, annotate, solver_parameters)

            # Call user defined callback
            if postsolver_callback is not None:
                postsolver_callback(config, state_new)

            # After the timestep solve, update state
            state.assign(state_new)
            if cache_forward_state:
                # Save state for initial guess cache
                log(INFO, "Cache initial guess for time %f." % t)
                if not self.state_cache.has_key(t):
                    self.state_cache[t] = Function(state_new.function_space())
                self.state_cache[t].assign(state_new, annotate=False)

            # Set the control function for the upcoming timestep.
            if turbine_field:
                if type(turbine_field) == list:
                    tf.assign(turbine_field[timestep])
                else:
                    tf.assign(turbine_field)

            if params["dump_period"] > 0 and step % params["dump_period"] == 0:
                log(INFO, "Write state to disk...")
                writer.write(state)

            if functional is not None:
                if not (functional_final_time_only and t < float(params["finish_time"])):
                    if steady_state or functional_final_time_only or functional_quadrature_degree == 0:
                        quad = 1.0
                    elif t >= float(params["finish_time"]):
                        quad = 0.5 * dt
                    else:
                        quad = 1.0 * dt

                    j += quad * assemble(functional.Jt(state, tf))
                    if params["print_individual_turbine_power"]:
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
            adj_inc_timestep(time=t, finished=(not t < float(params["finish_time"])))
        log(INFO, "End of time loop.")

        # Write the turbine positions, power extraction and friction to a .csv file named turbine_info.csv
        if params['print_individual_turbine_power']:
            f = config.params['base_path'] + os.path.sep + "iter_" + str(config.optimisation_iteration) + '/'
            # Save the very first result in a different file
            if config.optimisation_iteration == 0 and not os.path.isfile(f):
                f += 'initial_turbine_info.csv'
            else:
                f += 'turbine_info.csv'

            output_turbines = open(f, 'w')
            for i in range(0, len(individual_contribution_list), 5):
                print >> output_turbines, '%s, %s, %s, %s, %s' % (individual_contribution_list[i], individual_contribution_list[i + 1], individual_contribution_list[i + 2], individual_contribution_list[i + 3], individual_contribution_list[i + 4])
            print 'Total of individual turbines is', sum(j_individual)

        if functional is not None:
            return j
