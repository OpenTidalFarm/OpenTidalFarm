import inspect
import numpy

class Mutator(object):
    """
    Mutator class
    """
    def __init__(self, population, mutation_type, mutation_probability):
        self.population = population

        # make a dictionary of class methods (key=method name)
        self.valid_types = dict(inspect.getmembers(self, inspect.ismethod))

        if mutation_type not in self.valid_types:
            raise KeyError("%s is not a valid mutation type" % mutation_type)
        else:
            self.mutation_probability = mutation_probability
            # number of bits to mutate
            self.mutation_number = int(round(self.mutation_probability*
                                             self.population._population_size*
                                             self.population._n_turbines*2))
            self.mutation_type = mutation_type
            self.mutate = self.valid_types[self.mutation_type]


    def normal(self):
        """
        Mutates bits using a uniform distribution
        """
        indices_to_mutate = numpy.random.choice(
                                    range(self.population._population_size*
                                          self.population._n_turbines*2),
                                    self.mutation_number, replace=False)
        chromosomes_to_update = []
        for i in indices_to_mutate:
            # get a chromosome and gene indices
            c_i, g_i = divmod(i, self.population._n_turbines*2)
            chromosomes_to_update.append(c_i)
            # limits are ordered [xmin, ymin, xmax, ymax], so get correct limits
            lower = self.population._limits[(g_i%2)]
            upper = self.population._limits[(g_i%2)+2]
            self.population.population[c_i].turbine[g_i] = ((upper-lower)*
                                                      numpy.random.random() +
                                                      lower)
        # to_mutate contains reference so no need to return the mutated
        # chromosomes, however, return the indices of the mutated turbines so we
        # know which chromosomes to re-evaluate the fitness of
        return chromosomes_to_update


    def fitness_proportionate(self):
        """
        Variable rate mutation -- the fittest chromosomes are less likely to be
        mutated
        """
        max_probability = self.mutation_probability
        # get a copy of fitnesses and normalize
        fitnesses = numpy.array(self.population.get_fitnesses())
        max_fitness = max(fitnesses)
        fitnesses /= max_fitness
        # get a list of mutation probabilities
        mutate = [(1-f)*max_probability for f in fitnesses]
        mutated_indices = set([])
        for i in range(self.population._population_size):
            for j in range(self.population._n_turbines*2):
                beta = numpy.random.random()
                # mutate
                if beta < mutate[i]:
                    mutated_indices.add(i)
                    lower = self.population._limits[(j%2)]
                    upper = self.population._limits[(j%2)+2]
                    self.population.population[i].turbine[j] = ((upper-lower)*
                                                                beta + lower)
                # dont mutate
                else:
                    pass
        return mutated_indices
