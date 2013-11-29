from optimiser import GeneticOptimisation

def maximize_genetic(optimisation_problem):
    """
    Takes a GeneticOptimisation instance, runs the problem and returns the
    turbine positions (as tuples) for the best solution
    """
    # check if optimisation_problem is a GeneticOptimisation instance
    if not isinstance(optimisation_problem, GeneticOptimisation):
        raise RuntimeError("Must be a GeneticOptimisation instance")
    else:
        optimisation_problem.run()
        final_turbines = optimisation_problem.get_turbine_pos()
        turbines = []
        for i in range(len(final_turbines)/2):
            turbines.append((final_turbines[2*i], final_turbines[2*i+1]))
        return turbines
