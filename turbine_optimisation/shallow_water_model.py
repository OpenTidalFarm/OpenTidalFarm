import solver_benchmark 
import helpers
import numpy
import sys
from dolfin import *
from dolfin_adjoint import *
from helpers import info, info_green, info_red, info_blue

def sw_solve(config, state, turbine_field=None, functional=None, annotate=True, u_source = None):
    '''Solve the shallow water equations with the parameters specified in params.
       Options for linear_solver and preconditioner are: 
        linear_solver: lu, cholesky, cg, gmres, bicgstab, minres, tfqmr, richardson
        preconditioner: none, ilu, icc, jacobi, bjacobi, sor, amg, additive_schwarz, hypre_amg, hypre_euclid, hypre_parasails, ml_amg
    '''

    ############################### Setting up the equations ###########################

    # Define variables for all used parameters
    ds = config.domain.ds
    params = config.params

    # To begin with, check if the provided parameters are valid
    params.check()

    theta = params["theta"]
    dt = params["dt"]
    g = params["g"]
    depth = params["depth"]
    # Reset the time
    params["current_time"] = params["start_time"]
    t = params["current_time"]
    quadratic_friction = params["quadratic_friction"]
    include_advection = params["include_advection"]
    include_diffusion = params["include_diffusion"]
    diffusion_coef = params["diffusion_coef"]
    newton_solver = params["newton_solver"] 
    picard_relative_tolerance = params["picard_relative_tolerance"]
    picard_iterations = params["picard_iterations"]
    run_benchmark = params["run_benchmark"]
    solver_exclude = params["solver_exclude"]
    linear_solver = params["linear_solver"]
    preconditioner = params["preconditioner"]
    bctype = params["bctype"]
    strong_bc = params["strong_bc"]
    free_slip_on_sides = params["free_slip_on_sides"]
    steady_state = params["steady_state"]
    functional_final_time_only = params["functional_final_time_only"]
    is_nonlinear = (include_advection or quadratic_friction)
    
    # Print out an estimation of the Reynolds number 
    if include_diffusion and diffusion_coef>0:
      reynolds = params["turbine_x"]*2./diffusion_coef
    else:
      reynolds = "oo"
    info("Expected Reynolds number is roughly (assumes velocity is 2): %s" % str(reynolds))

    # Take care of the steady state case
    if steady_state:
        dt = 1.
        params["finish_time"] = params["start_time"] + dt/2
        theta = 1.

    # Define test functions
    (v, q) = TestFunctions(config.function_space)

    # Define functions
    state_new = Function(config.function_space, name="New_state")  # solution of the next timestep 
    state_nl = Function(config.function_space, name="Best_guess_state")  # the last computed state of the next timestep, used for the picard iteration

    # Split mixed functions
    if is_nonlinear and newton_solver:
      u, h = split(state_new) 
    else:
      u, h = TrialFunctions(config.function_space) 
    u0, h0 = split(state)
    u_nl, h_nl = split(state_nl)

    # Create initial conditions and interpolate
    state_new.assign(state, annotate=annotate)

    # u_(n+theta) and h_(n+theta)
    u_mid = (1.0-theta)*u0 + theta*u
    h_mid = (1.0-theta)*h0 + theta*h

    # If a picard iteration is used we need an intermediate state 
    if is_nonlinear and not newton_solver:
      u_nl, h_nl = split(state_nl)
      state_nl.assign(state, annotate=annotate)
      u_mid_nl = (1.0-theta)*u0 + theta*u_nl

    # The normal direction
    n = FacetNormal(config.function_space.mesh())

    # Mass matrix

    M = inner(v, u) * dx
    M += inner(q, h) * dx
    M0 = inner(v, u0) * dx
    M0 += inner(q, h0) * dx

    # Divergence term.
    Ct_mid = -depth * inner(u_mid, grad(q))*dx
    #+inner(avg(u_mid),jump(q,n))*dS # This term is only needed for dg element pairs

    if bctype == 'dirichlet':
      if steady_state:
          raise ValueError, "Can not use a time dependent boundary condition for a steady state simulation"
      # The dirichlet boundary condition on the left hand side 
      ufl = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0", "0"), eta0=params["eta0"], g=g, depth=depth, t=t, k=params["k"])
      bc_contr = - depth * dot(ufl, n) * q * ds(1)

      # The dirichlet boundary condition on the right hand side
      bc_contr -= depth * dot(ufl, n) * q * ds(2)

      # We enforce a no-normal flow on the sides by removing the surface integral. 
      # bc_contr -= dot(u_mid, n) * q * ds(3)

    elif bctype == 'flather':
      if steady_state:
          raise ValueError, "Can not use a time dependent boundary condition for a steady state simulation"
      # The Flather boundary condition on the left hand side 
      ufl = Expression(("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0", "0"), eta0=params["eta0"], g=g, depth=depth, t=t, k=params["k"])
      bc_contr = - depth * dot(ufl, n) * q * ds(1)
      Ct_mid += sqrt(g*depth)*inner(h_mid, q)*ds(1)

      # The contributions of the Flather boundary condition on the right hand side
      Ct_mid += sqrt(g*depth)*inner(h_mid, q)*ds(2)

    elif bctype == 'strong_dirichlet':
        # Do not replace anything in the surface integrals as the strong Dirichlet Boundary condition will do that
        bc_contr = -depth * dot(u_mid, n) * q * ds(1)
        bc_contr -= depth * dot(u_mid, n) * q * ds(2)
        if not free_slip_on_sides:
            bc_contr -= depth * dot(u_mid, n) * q * ds(3)

    else:
        info_red("Unknown boundary condition type: %s" % bctype)
        sys.exit(1)

    # Pressure gradient operator
    C_mid = g * inner(v, grad(h_mid)) * dx
    #+inner(avg(v),jump(h_mid,n))*dS # This term is only needed for dg element pairs

    # Bottom friction
    class FrictionExpr(Expression):
        def eval(self, value, x):
           value[0] = params["friction"] 

    friction = FrictionExpr()
    if turbine_field:
      friction += turbine_field

    # Friction term
    # With a newton solver we can simply use a non-linear form
    if quadratic_friction and newton_solver:
      R_mid = friction / depth * dot(u_mid, u_mid)**0.5 * inner(u_mid, v) * dx 
    # With a picard iteration we need to linearise using the best guess
    elif quadratic_friction and not newton_solver:
      R_mid = friction / depth * dot(u_mid_nl, u_mid_nl)**0.5 * inner(u_mid, v) * dx 
    # Use a linear drag
    else:
      R_mid = friction / depth * inner(u_mid, v) * dx 

    # Advection term 
    # With a newton solver we can simply use a quadratic form
    if include_advection and newton_solver:
      Ad_mid = inner(dot(grad(u_mid), u_mid), v)*dx
    # With a picard iteration we need to linearise using the best guess
    if include_advection and not newton_solver:
      Ad_mid = inner(dot(grad(u_mid), u_mid_nl), v)*dx

    if include_diffusion:
      # Check that we are not using a DG velocity function space, as the facet integrals are not implemented.
      if "Discontinuous" in str(config.function_space.split()[0]):
        raise NotImplementedError, "The diffusion term for discontinuous elements is not implemented yet."
      D_mid = diffusion_coef*inner(grad(u_mid), grad(v))*dx

    # Create the final form
    G_mid = C_mid + Ct_mid + R_mid 
    # Add the advection term
    if include_advection:
        G_mid += Ad_mid
    # Add the diffusion term
    if include_diffusion:
        G_mid += D_mid
    # Add the source term
    if u_source:
        G_mid -= inner(u_source, v)*dx 
    F = dt * G_mid - dt * bc_contr
    if not steady_state:
        F += M - M0 

    # Preassemble the lhs if possible
    use_lu_solver = (linear_solver == "lu") 
    if not quadratic_friction and not include_advection:
        lhs_preass = assemble(dolfin.lhs(F))
        # Precompute the LU factorisation 
        if use_lu_solver:
          info("Computing the LU factorisation for later use ...")
          if bctype == 'strong_dirichlet':
              raise NotImplementedError, "Strong boundary condition and reusing LU factorisation is currently not implemented"
          lu_solver = LUSolver(lhs_preass)
          lu_solver.parameters["reuse_factorization"] = True

    solver_parameters = {"linear_solver": linear_solver, "preconditioner": preconditioner}

    ############################### Perform the simulation ###########################

    writer = helpers.StateWriter(config)
    writer.write(state)
    
    step = 0    

    if functional is not None:
        if steady_state or functional_final_time_only:
            j = 0.
            if params["print_individual_turbine_power"]:
	        j_individual = [0] * len(params["turbine_pos"])

            djdm = 0.
        else:
            quad = 0.5
            j =  dt * quad * assemble(functional.Jt(state)) 
            if params["print_individual_turbine_power"]:
                j_individual = []
                for i in range(len(params["turbine_pos"])):
	            j_individual.append(dt * quad * assemble(functional.Jt_individual(state, i))) 
              
            djdm = dt * quad * numpy.array([assemble(f) for f in functional.dJtdm(state)])

    adjointer.time.start(t)
    while (t < params["finish_time"]):
        t += dt
        params["current_time"] = t

        # Update bc's 
        if bctype == "strong_dirichlet":
            strong_bc.update_time(t)
        else:
            ufl.t=t-(1.0-theta)*dt         
        # Update source term
        if u_source:
            u_source.t = t-(1.0-theta)*dt          
        step+=1
        
        # Solve non-linear system with a Newton sovler
        if is_nonlinear and newton_solver:
          # Use a Newton solver to solve the nonlinear problem.
          #solver_parameters["linear_solver"] = "gmres"
          #solver_parameters["linear_solver"] = "superlu_dist"
          #solver_parameters["preconditioner"] = "ilu" # does not work in parallel
          #solver_parameters["preconditioner"] = "amg" 
          #solver_parameters["linear_solver"] = "mumps"
          solver_parameters["newton_solver"] = {}
          solver_parameters["newton_solver"]["convergence_criterion"] = "incremental"
          solver_parameters["newton_solver"]["relative_tolerance"] = 1e-16
          if bctype == 'strong_dirichlet':
              solver_benchmark.solve(F == 0, state_new, bcs = strong_bc.bcs, solver_parameters = solver_parameters, annotate=annotate, benchmark = run_benchmark, solve = solve, solver_exclude = solver_exclude)
          else:
              solver_benchmark.solve(F == 0, state_new, solver_parameters = solver_parameters, annotate=annotate, benchmark = run_benchmark, solve = solve, solver_exclude = solver_exclude)

        # Solve non-linear system with a Picard iteration
        elif is_nonlinear:
          # Solve the problem using a picard iteration
          iter_counter = 0
          while True:
            if bctype == 'strong_dirichlet':
                solver_benchmark.solve(dolfin.lhs(F) == dolfin.rhs(F), state_new, bcs = strong_bc.bcs, solver_parameters = solver_parameters, annotate=annotate, benchmark = run_benchmark, solve = solve, solver_exclude = solver_exclude)
            else:
                solver_benchmark.solve(dolfin.lhs(F) == dolfin.rhs(F), state_new, solver_parameters = solver_parameters, annotate=annotate, benchmark = run_benchmark, solve = solve, solver_exclude = solver_exclude)
            iter_counter += 1
            if iter_counter > 0:
              relative_diff = abs(assemble( inner(state_new-state_nl, state_new-state_nl) * dx ))/assemble( inner(state_new, state_new) * dx )
              info_blue("Picard iteration " + str(iter_counter) + " relative difference: " + str(relative_diff))

              if relative_diff < picard_relative_tolerance:
                info("Picard iteration converged after " + str(iter_counter) + " iterations.")
                break
              elif iter_counter >= picard_iterations:
                info_red("Picard iteration reached maximum number of iterations (" + str(picard_iterations) + ") with a relative difference of " + str(relative_diff) + ".")
                break

            state_nl.assign(state_new)

        # Solve linear system with preassembled matrices 
        else:
            rhs_preass = assemble(dolfin.rhs(F))
            # Apply dirichlet boundary conditions
            if bctype == 'strong_dirichlet':
                [bc.apply(lhs_preass, rhs_preass) for bc in strong_bc.bcs]
            if use_lu_solver:
                info("Using a LU solver to solve the linear system.")
                lu_solver.solve(state.vector(), rhs_preass, annotate=annotate)
            else:
                state_tmp = Function(state.function_space(), name="TempState")
                solver_benchmark.solve(lhs_preass, state_new.vector(), rhs_preass, solver_parameters["linear_solver"], solver_parameters["preconditioner"], annotate=annotate, benchmark = run_benchmark, solve = solve, solver_exclude = solver_exclude)

        # After the timestep solve, update state
        state.assign(state_new)

        if step%params["dump_period"] == 0:
            writer.write(state)

        if functional is not None:
            if not (functional_final_time_only and t < params["finish_time"]):
                if steady_state or functional_final_time_only:
                    quad = 1.0
                elif t >= params["finish_time"]:
                    quad = 0.5 * dt
                else:
                    quad = 1.0 * dt 

                j += quad * assemble(functional.Jt(state)) 
                if params["print_individual_turbine_power"]:
                    for i in range(len(params["turbine_pos"])):
	                j_individual[i] += dt * quad * assemble(functional.Jt_individual(state, i))

                djtdm = numpy.array([assemble(f) for f in functional.dJtdm(state)])
                djdm += quad * djtdm

        # Increase the adjoint timestep
        adj_inc_timestep(time=t, finished = not t < params["finish_time"])

    if params["print_individual_turbine_power"]:
        print "Individual power contributions of the turbines: ", j_individual
    if functional is not None:
        return j, djdm 

