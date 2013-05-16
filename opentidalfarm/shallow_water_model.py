import solver_benchmark 
import helpers
import numpy
import sys
from dolfin import *
from dolfin_adjoint import *
from helpers import info, info_green, info_red, info_blue, print0
import ufl

# Some global variables for plotting the thurst plot - just for testing purposes 
us = []
thrusts = []
thrusts_est = []

distance_to_upstream = 1.*20

def uflmin(a, b):
    return conditional(lt(a, b), a, b)
def uflmax(a, b):
    return conditional(gt(a, b), a, b)
def norm_approx(u, alpha=1e-4):
   # A smooth approximation to ||u||
   return sqrt(inner(u, u)+alpha**2)

def smooth_uflmin(a, b, alpha=1e-8):
    return a - (norm_approx(a-b, alpha=alpha) + a-b)/2 

def upstream_u_implicit_equation(config, tf, u, up_u, o, up_u_adv, o_adv):
        ''' Returns the implicit equations that compute the turbine upstream velocities ''' 

        def advect(u, up_u_adv, o_adv):

            theta = 1.0
            uh = Constant(1-theta)*norm_approx(u) + Constant(theta)*up_u_adv
            F = (inner(up_u_adv-norm_approx(u), o_adv) + Constant(distance_to_upstream)/norm_approx(uh)*inner(dot(grad(uh), u), o_adv))*dx 
            #F = (inner(up_u_adv-norm_approx(u), o_adv) + Constant(distance_to_upstream)*inner(dot(grad(uh), Constant((1, 0))), o_adv))*dx 

            return F

        def smooth(up_u_adv, up_u, o):
            # Calculate averaged velocities

            # Define the indicator function of the turbine support 
            chi = ufl.conditional(ufl.gt(tf, 0), 1, 0)

            c_diff = Constant(1e6)
            F1 = chi*(inner(up_u-up_u_adv, o) + c_diff*inner(grad(up_u), grad(o)))*dx 
            invchi = 1-chi
            F2 = inner(invchi*up_u, o)*dx 
            F = F1 + F2

            return F

        up_u_eqs = advect(u, up_u_adv, o_adv) + smooth(up_u_adv, up_u, o)
        return up_u_eqs

def upstream_u_equation(config, tf, u, up_u, o):
        ''' Returns the equation that computes the turbine upstream velocities ''' 

        # The equations underpredict the upstream velocity which is corrected with this factor
        correction_factor = Constant(1.34)

        def smooth(u, up_u, o):
            # Calculate averaged velocities

            # Define the indicator function of the turbine support 
            chi = ufl.conditional(ufl.gt(tf, 0), 1, 0)

            # Solve the Helmholtz equation in each turbine area to obtain averaged velocity values
            c_diff = Constant(1e6)
            F1 = chi*(inner(up_u-norm_approx(u), o) + Constant(distance_to_upstream)/norm_approx(u)*(inner(dot(grad(norm_approx(u)), u), o) + c_diff*inner(grad(up_u), grad(o))))*dx 
            invchi = 1-chi
            F2 = inner(invchi*up_u, o)*dx 
            F = F1 + F2

            return F

        up_u_eq = smooth(u, 1./correction_factor*up_u, o) 
        return up_u_eq

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
    turbine_thrust_parametrisation = params["turbine_thrust_parametrisation"]
    implicit_turbine_thrust_parametrisation = params["implicit_turbine_thrust_parametrisation"]

    if implicit_turbine_thrust_parametrisation:
        function_space = config.function_space_2enriched
    elif turbine_thrust_parametrisation:
        function_space = config.function_space_enriched
    else:
        function_space = config.function_space
    
    # Take care of the steady state case
    if steady_state:
        dt = 1.
        params["finish_time"] = params["start_time"] + dt/2
        theta = 1.

    # Define test functions
    if implicit_turbine_thrust_parametrisation:
        v, q, o, o_adv = TestFunctions(function_space)
    elif turbine_thrust_parametrisation:
        v, q, o = TestFunctions(function_space)
    else:
        v, q = TestFunctions(function_space)

    # Define functions
    state_new = Function(function_space, name="New_state")  # solution of the next timestep 
    state_nl = Function(function_space, name="Best_guess_state")  # the last computed state of the next timestep, used for the picard iteration

    if not newton_solver and (turbine_thrust_parametrisation or implicit_turbine_thrust_parametrisation):
        raise NotImplementedError, "Thrust turbine representation does currently only work with the newton solver." 

    # Split mixed functions
    if is_nonlinear and newton_solver:
        if implicit_turbine_thrust_parametrisation:
            u, h, up_u, up_u_adv = split(state_new) 
        elif turbine_thrust_parametrisation:
            u, h, up_u = split(state_new) 
        else:
            u, h = split(state_new) 
    else:
        u, h = TrialFunctions(function_space) 

    if implicit_turbine_thrust_parametrisation:
        u0, h0, up_u0, up_u_adv0 = split(state)
    elif turbine_thrust_parametrisation:
        u0, h0, up_u0 = split(state)
    else:
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
    n = FacetNormal(function_space.mesh())

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
      ufl = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), eta0=params["eta0"], g=g, depth=depth, t=t, k=params["k"])
      bc_contr = - depth * dot(ufl, n) * q * ds(1)

      # The dirichlet boundary condition on the right hand side
      bc_contr -= depth * dot(ufl, n) * q * ds(2)

      # We enforce a no-normal flow on the sides by removing the surface integral. 
      # bc_contr -= dot(u_mid, n) * q * ds(3)

    elif bctype == 'flather':
      if steady_state:
          raise ValueError, "Can not use a time dependent boundary condition for a steady state simulation"
      # The Flather boundary condition on the left hand side 
      ufl = Expression(("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"), eta0=params["eta0"], g=g, depth=depth, t=t, k=params["k"])
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
        if type(turbine_field) == list:
           tf = project(turbine_field[0], turbine_field[0].function_space(), name="turbine_friction")
        else:
           tf = project(turbine_field, turbine_field.function_space(), name="turbine_friction")

        if not turbine_thrust_parametrisation and not implicit_turbine_thrust_parametrisation:
            friction += tf
        else:
            print "Adding thrust force"
            # Compute the upstream velocities

            if implicit_turbine_thrust_parametrisation:
                up_u_eq = upstream_u_implicit_equation(config, tf, u, up_u, o, up_u_adv, o_adv)
            elif turbine_thrust_parametrisation:
                up_u_eq = upstream_u_equation(config, tf, u, up_u, o)

            def thrust_force(up_u, min=smooth_uflmin):
                ''' Returns the thrust force for a given upstream velcocity ''' 
                # Now apply a pointwise transformation based on the interpolation of a loopup table 
                c_T_coeffs = [0.08344535,  -1.42428216, 9.13153605, -26.19370168, 28.8752054]
                c_T_coeffs.reverse()
                c_T = min(0.88, sum([c_T_coeffs[i]*up_u**i for i in range(len(c_T_coeffs))]))

                # The amount of forcing we want to apply
                turbine_radius = 15.
                A_c = pi*Constant(turbine_radius**2) # Turbine cross section
                f = 0.5*c_T*up_u**2*A_c
                return f

    
            # Apply the force in the opposite direction of the flow
            f_dir = -thrust_force(up_u)*u/norm_approx(u, alpha=1e-6)
            # Distribute this force over the turbine area
            thrust = inner(f_dir*tf/(Constant(config.turbine_cache.turbine_integral())*config.params["depth"]), v)*dx
        
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
      if "Discontinuous" in str(function_space.split()[0]):
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
    # Add the time term
    if not steady_state:
        F += M - M0 

    if turbine_thrust_parametrisation or implicit_turbine_thrust_parametrisation:
        F += up_u_eq
        if turbine_field:
            F -= thrust

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

    # Do some parameter checking:
    if "dynamic_turbine_friction" in params["controls"]:
       if len(config.params["turbine_friction"]) != (params["finish_time"]-t)/dt+1:
          print "You control the turbine friction dynamically, but your turbine friction parameter is not an array of length 'number of timesteps' (here: %i)." % ((params["finish_time"]-t)/dt+1)
          import sys; sys.exit()

    ############################### Perform the simulation ###########################

    if params["dump_period"] > 0:
        writer = helpers.StateWriter(config)
        print0("Writing state to disk...")
        writer.write(state)
    
    step = 0    

    if functional is not None:
        if steady_state or functional_final_time_only:
            j = 0.

        else:
            quad = 0.5
            j =  dt * quad * assemble(functional.Jt(state, tf))
              
    print0("Starting time loop...")
    adjointer.time.start(t)
    timestep = 0
    while (t < params["finish_time"]):
        timestep += 1
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
          #solver_parameters["newton_solver"]["error_on_nonconvergence"] = False
          solver_parameters["newton_solver"]["maximum_iterations"] = 20 
          solver_parameters["newton_solver"]["convergence_criterion"] = "incremental"
          solver_parameters["newton_solver"]["relative_tolerance"] = 1e-16

          if bctype == 'strong_dirichlet':
              solver_benchmark.solve(F == 0, state_new, bcs = strong_bc.bcs, solver_parameters = solver_parameters, annotate=annotate, benchmark = run_benchmark, solve = solve, solver_exclude = solver_exclude)
          else:
              solver_benchmark.solve(F == 0, state_new, solver_parameters = solver_parameters, annotate=annotate, benchmark = run_benchmark, solve = solve, solver_exclude = solver_exclude)

          if turbine_thrust_parametrisation or implicit_turbine_thrust_parametrisation:
              print "Inflow velocity: ", u[0]((10, 160))
              print "Estimated upstream velocity: ", up_u((640./3, 160))
              print "Expected thrust force: ", thrust_force(u[0]((10, 160)), min=min)((0))
              print "Total amount of thurst force applied: ", assemble(inner(Constant(1), thrust_force(up_u)*tf/config.turbine_cache.turbine_integral())*dx)

              us.append(u[0]((10, 160)))
              thrusts.append(thrust_force(u[0]((10, 160)))((0)))
              thrusts_est.append(assemble(inner(Constant(1), thrust_force(up_u)*tf/config.turbine_cache.turbine_integral())*dx))

              import matplotlib.pyplot as plt
              plt.clf()
              plt.plot(us, thrusts, label="Analytical")
              plt.plot(us, thrusts_est, label="Approximated")
              plt.legend(loc=2)
              plt.savefig("thrust_plot.pdf", format='pdf')

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
                state_tmp = Function(function_space, name="TempState")
                solver_benchmark.solve(lhs_preass, state_new.vector(), rhs_preass, solver_parameters["linear_solver"], solver_parameters["preconditioner"], annotate=annotate, benchmark = run_benchmark, solve = solve, solver_exclude = solver_exclude)

        # After the timestep solve, update state
        state.assign(state_new)

        # Set the control function for the upcoming timestep.
        if turbine_field:
            if type(turbine_field) == list:
               tf.assign(turbine_field[timestep])
            else:

               tf.assign(turbine_field)

        if params["dump_period"] > 0 and step%params["dump_period"] == 0:
            print0("Writing state to disk...")
            writer.write(state)

        if functional is not None:
            if not (functional_final_time_only and t < params["finish_time"]):
                if steady_state or functional_final_time_only:
                    quad = 1.0
                elif t >= params["finish_time"]:
                    quad = 0.5 * dt
                else:
                    quad = 1.0 * dt 

                j += quad * assemble(functional.Jt(state, tf))

        # Increase the adjoint timestep
        adj_inc_timestep(time=t, finished = not t < params["finish_time"])
        print0("New timestep t = %f" % t)
    print0("Ending time loop.")

    if functional is not None:
        return j

