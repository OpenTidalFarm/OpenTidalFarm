class ParameterDictionary(dict):
    '''Parameter dictionary. This subclasses dict so defaults can be set.'''

    def __init__(self, dic={}):
        # Apply dic after defaults so as to overwrite the defaults
        for key, val in dic.iteritems():
            self[key] = val

        self.required = {
            'verbose': 'output verbosity',
            'dt': 'timestep',
            'theta': 'the implicitness for the time discretisation',
            'start_time': 'start time',
            'current_time': 'current time',
            'finish_time': 'finish time',
            'steady_state': 'steady state simulation',
            'functional_final_time_only': 'if the functional should be evaluated at the final time only (used if the time stepping is used to converge to a steady state)',
            'functional_quadrature_degree': 'quadrature degree of the functional integral evaluation',
            'dump_period': 'dump period in timesteps; use 0 to deactivate disk outputs',
            'bctype': 'type of boundary condition to be applied',
            'strong_bc': 'list of strong dirichlet boundary conditions to be applied',
            'flather_bc_expr': 'dolfin.Expression describing the flather boundary condition values',
            'u_weak_dirichlet_bc_expr': 'dolfin.Expression describing the weak Dirichlet boundary condition values for velocity',
            'eta_weak_dirichlet_bc_expr': 'dolfin.Expression describing the weak Dirichlet boundary condition values for free surface height',
            'free_slip_on_sides': 'apply free slip boundary conditions on the sides (id=3)',
            'initial_condition': 'initial condition function',
            'include_advection': 'advection term on',
            'include_diffusion': 'diffusion term on',
            'include_time_term': 'time term is included',
            'diffusion_coef': 'diffusion coefficient',
            'cost_coef': 'multiplicator that determines the cost per turbine friction',
            'linear_divergence': 'use the depth at rest as an approximation for the full depth in the shallow water equations',
            'depth': 'water depth at rest',
            'g': 'graviation',
            'quadratic_friction': 'quadratic friction',
            'friction': 'friction term on',
            'turbine_parametrisation': 'parametrisation of the turbines. If its value is "individual" then the turbines are resolved individually, if "smooth" then the turbines are represented as an average friction over the site area',
            'turbine_pos': 'list of turbine positions',
            'turbine_x': 'turbine extension in the x direction',
            'turbine_y': 'turbine extension in the y direction',
            'turbine_friction': 'turbine friction',
            'turbine_thrust_parametrisation': 'parametrise the turbine based on speed/thrust and speed/power functions. If False, the turbines are parametrised as increased friction.',
            'implicit_turbine_thrust_parametrisation': 'implicitly parametrise the turbine based on speed/thrust and speed/power functions. If False, the turbines are parametrised as increased friction.',
            'rho': 'the density of the fluid',
            'controls': 'a list of the control variables. Valid list values: "turbine_pos" for the turbine position, "turbine_friction" for the friction of the turbine',
            'newton_solver': 'newton solver instead of a picard iteration',
            'postsolver_callback': 'a function which is called after each solve',
            'solver_parameters': 'a dictionary containing the solver settings. Must be compatible to DOLFIN\'s solve interface.',
            'picard_relative_tolerance': 'relative tolerance for the picard iteration',
            'picard_iterations': 'maximum number of picard iterations',
            'run_benchmark': 'benchmark to compare different solver/preconditioner combinations',
            'solver_exclude': 'solvers/preconditioners to be excluded from the benchmark',
            'automatic_scaling': 'activates the initial automatic scaling of the functional',
            'automatic_scaling_multiplier': 'defines the multiplier that determines the initial gradient length (= multiplier * turbine size)',
            'print_individual_turbine_power': 'print out the power output of each individual turbine',
            'output_turbine_power': 'output the power generation of the individual turbines',
            'save_checkpoints': 'automatically store checkpoints after each optimisation iteration',
            'cache_forward_state': 'caches the forward state for all timesteps and reuses them as initial guess for the next optimisation iteration',
            'base_path': 'root directory for output',
            'nonlinear_solver': 'callback to solve the nonlinear problem. Called with callback(F, z, bcs, annotate, solver_parameters).',
            'revolve_parameters': '(strategy, snaps_on_disk, snaps_in_ram, verbose)',
             }

    def check(self):
        # First check that no parameters are missing
        for key, error in self.required.iteritems():
            if key not in self:
                raise KeyError('Missing parameter: ' + key + '. ' + 'This is used to set the ' + error + '.')
        # Then check that no parameter is too much (as this is likely to be a mistake!)
        diff = set(self.keys()) - set(self.required.keys())
        if len(diff) > 0:
            raise KeyError('Configuration has too many parameters: ' + str(diff))
