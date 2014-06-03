from selector import Selector
from mutator import Mutator
from crossover import Crossover
import copy

class Generate(object):
    """
    A generator class for genetic algorithms
    """
    def __init__(self, population, survival_rate, crossover_type, mutation_type,
                 selection_type, mutation_probability, selection_options):
        
        self.population = population
        # initialize the selector, mutator and crossover operators
        self.selector = Selector(self.population, survival_rate, selection_type,
                                 selection_options)
        self.mutator = Mutator(self.population, mutation_type,
                               mutation_probability)
        # TODO: selection proportion -- proportion of population to keep
        self.crossover = Crossover(self.population, crossover_type,
                                   survival_rate, selection_type,
                                   selection_options)
        
    def generate(self):
        # perform a generation by first selecting the parent indices
        parent_indices = self.selector.select()
        # crossover parents to generate children
        children_turbines = self.crossover.mate(parent_indices)
        # update the population
        n_parents = len(parent_indices)
        n_children = len(children_turbines)
        # check the population size is correct
        try:
            assert(n_children + n_parents == self.population._population_size)
        except:
            raise ValueError("New generation size is not equal to previous "
                             "generation")
        # parents should be unchanged, so only update the turbines in the
        # remaining chromosomes
        pop_set = set(range(self.population._population_size))
        pop_set.difference_update(set(parent_indices))
        to_update = list(pop_set)
        for i in range(n_children):
            # calling the chromosome object ubdates the turbine positions and
            # reevaluates its fitness
            self.population.population[to_update[i]](children_turbines[i])
        # mutate the population and get the indices of the mutated chromosomes
        mutated_indices = self.mutator.mutate()
        for i in mutated_indices:
            self.population.population[i].update_fitness()
        # update the population fitness
        self.population.current_fitnesses = self.population.update_fitnesses()

        # find the fittest individual and check against the global fittest
        fitnesses = self.population.get_sorted_fitnesses()
        if fitnesses[0][1] > self.population.global_maximum[1]:
            fittest = self.population.population[fitnesses[0][0]]
            # make a deepcopy just in case we lose the object
            turbine = copy.deepcopy(fittest.turbine)
            fitness = copy.deepcopy(fittest.get_fitness())
            self.population.global_maximum = (turbine, fitness)
