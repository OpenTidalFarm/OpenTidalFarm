import dolfin
import ufl
import operator

def solver_parameters(solver_exclude, preconditioner_exclude):
    linear_solver_set = ["lu"] 
    linear_solver_set += [e[0] for e in dolfin.krylov_solver_methods()]
    preconditioner_set = [e[0] for e in dolfin.krylov_solver_preconditioners()]

    solver_parameters_set = []
    for l in linear_solver_set:
        if l in solver_exclude:
            continue
        for p in preconditioner_set:
            if p in preconditioner_exclude:
                continue
            if (l == "lu" or l == "default") and p != "none":
                continue
            solver_parameters_set.append({"linear_solver": l, "preconditioner": p})
    return solver_parameters_set

def print_benchmark_report(solver_timings, failed_solvers):
        # Let's analyse the result of the benchmark test:
        solver_timings = sorted(solver_timings.iteritems(), key=operator.itemgetter(1)) 
        failed_solvers = sorted(failed_solvers.iteritems(), key=operator.itemgetter(1)) 

        dolfin.info_blue("***********************************************") 
        dolfin.info_blue("********** Solver benchmark results: **********")
        dolfin.info_blue("***********************************************") 
        for solver, timing in solver_timings:
            dolfin.info_blue("%s: %.2f s" % (solver, timing))
        for solver, reason in failed_solvers:
            dolfin.info_red("%s: %s" % (solver, reason))

def replace_solver_settings(args, kwargs, parameters):
    ''' Replace the arguments of a solve call and replace the solver settings with the ones given in solver_settings. '''

    # The way how to set the solver settings depends on how the system is solved:
    #  Adaptive solve 
    if "tol" in kwargs:
        raise NotImplementedError, 'The benchmark solver is currently not implemented for adaptive solver calls.'

    # Variational problem solver 
    elif isinstance(args[0], ufl.classes.Equation):
        kwargs['solver_parameters'] = parameters

    # Default case: call the c++ solve routine
    else:
        args = args[0:3] + (parameters['linear_solver'], parameters['preconditioner'])

    return args, kwargs


def solve(*args, **kwargs):
    ''' This function overwrites the dolfin.solve function but provides additional functionality to benchmark 
        different solver/preconditioner settings. The arguments of equivalent to dolfin.solve except some (optional) additional parameters:
        - benchmark = [True, False]: If True, the problem will be solved with all different solver/precondition combinations and the results reported.
                                     If False, the problem is solved using the default solver settings.
        - solve: An optional function parameter that is called instead of dolfin.solve. This parameter is useful if dolfin.solve is overwritten by a custom solver routine.
        - solver_exclude: A list of solvers that are to be excluded from the benchmark.
        - preconditioner_exclude: A list of preconditioners that are to be excluded from the benchmark.

    '''

    # Retrieve the extended benchmark arguments.
    if kwargs.has_key('benchmark'):
        benchmark = kwargs.pop('benchmark')
    else:
        benchmark = False

    if kwargs.has_key('solve'):
        solve = kwargs.pop('solve')
    else:
        solve = dolfin.fem.solving.solve

    if kwargs.has_key('solver_exclude'):
        solver_exclude = kwargs.pop('solver_exclude')
    else:
        solver_exclude = [] 

    if kwargs.has_key('preconditioner_exclude'):
        preconditioner_exclude = kwargs.pop('preconditioner_exclude')
    else:
        preconditioner_exclude = [] 

    if benchmark: 
        dolfin.info_blue("Running solver benchmark...")
        solver_parameters_set = solver_parameters(solver_exclude, preconditioner_exclude)
        solver_timings = {}
        failed_solvers = {}
        ret = None

        # Perform the benchmark
        for parameters in solver_parameters_set:
            solver_failed = False
            # Replace the existing solver setting with the benchmark one's.
            new_args, new_kwargs = replace_solver_settings(args, kwargs, parameters) 

            # Solve the problem
            timer = dolfin.Timer("Solver benchmark")
            timer.start()
            try:
                ret = solve(*new_args, **new_kwargs)
            except RuntimeError as e:
                solver_failed = True
                if 'diverged' in e.message.lower():
                    failure_reason = 'diverged'
                else:
                    failure_reason = 'unknown'
                    from IPython.Shell import IPShellEmbed
                    ipshell = IPShellEmbed()
                    ipshell()
                pass
            timer.stop()

            # Save the result
            parameters_str = parameters["linear_solver"] + ", " + parameters["preconditioner"]
            if solver_failed:
                dolfin.info_red(parameters_str + ": solver failed.")
                failed_solvers[parameters_str] = failure_reason 
            else:
                dolfin.info(parameters_str + ": " + str(timer.value()) + "s.")
                solver_timings[parameters_str] = timer.value() 

        # Print the report
        print_benchmark_report(solver_timings, failed_solvers) 

    else:
        ret = solve(*args, **kwargs)
    return ret

