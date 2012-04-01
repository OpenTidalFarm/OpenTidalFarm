import dolfin

def solver_parameters(solver_exclude, preconditioner_exclude):
    linear_solver_set = ["lu", "gmres", "bicgstab", "minres", "tfqmr", "richardson"]
    preconditioner_set =["none", "ilu", "icc", "jacobi", "bjacobi", "sor", "amg", "additive_schwarz", "hypre_amg", "hypre_euclid", "hypre_parasails"]

    solver_parameters_set = []
    for l in linear_solver_set:
        if l in solver_exclude:
            continue
        for p in preconditioner_set:
            if p in preconditioner_exclude:
                continue
            if l == "lu" and p != "none":
                continue
            if l == "default" and p != "none":
                continue
            if l == "cholesky" and p != "none":
                continue
            if l == "gmres" and (p == "none" or p == "default"):
                continue
            solver_parameters_set.append({"linear_solver": l, "preconditioner": p})
    return solver_parameters_set

def print_benchmark_results(solver_benchmark_results):
        # Let's analyse the result of the benchmark test:
        def sortDict(dict):
            keys = dict.keys()
            keys.sort()
            return keys, [dict[key] for key in keys]

        times, solvers = sortDict(solver_benchmark_results)

        dolfin.info_blue("***********************************************") 
        dolfin.info_blue("********** Solver benchmark results: **********")
        dolfin.info_blue("***********************************************") 
        for i in range(len(times)):
            dolfin.info_blue("%s, %s: %.2f s" % (str(solvers[i]['linear_solver']), str(solvers[i]['preconditioner']), times[i]))

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
        solver_benchmark_results = {}

        # Perform the benchmark
        for parameters in solver_parameters_set:
            solver_failed = False
            timer = dolfin.Timer("Solver benchmark")
            timer.start()

            try:
                ret = solve(*args, **kwargs)
            except RunTimeError:
                solver_failed = True
                pass

            timer.stop()
            if solver_failed:
                dolfin.info_red(parameters["linear_solver"] + ", " + parameters["preconditioner"] + ": solver failed.")
            else:
                dolfin.info(parameters["linear_solver"] + ", " + parameters["preconditioner"] + ": " + str(timer.value()) + "s.")
            if not solver_failed:
                solver_benchmark_results[timer.value()] = parameters

        # Print the results
        print_benchmark_results(solver_benchmark_results) 

    else:
        ret = solve(*args, **kwargs)
    return ret

