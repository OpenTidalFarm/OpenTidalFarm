from chromosome import Chromosome, ChromosomeContainer
import operator
import copy

class Population(object):
    def __init__(self, config, population_size, number_of_turbines,
                 ambient_flow, wake_model_type, wake_model_parameters):

        self._population_size = population_size
        self._n_turbines = number_of_turbines
        self._flow = ambient_flow
        self._config = config
        self._limits = [self._config.domain.site_x_start,
                        self._config.domain.site_y_start,
                        self._config.domain.site_x_end,
                        self._config.domain.site_y_end]

        # initialize a chromosome container -- i.e. the wake model and other
        # shared parameters
        self.container = ChromosomeContainer(self._config,
                                             self._flow,
                                             self._limits,
                                             wake_model_type,
                                             wake_model_parameters)

        # initialize a population
        self.fitnesses = None
        self.population = self._initialize_population(wake_model_type,
                                                      wake_model_parameters)
        self.update_fitnesses()

        fittest_index = self.get_sorted_fitnesses()[0][0]
        fittest_chromosome = self.population[fittest_index]
        self.global_maximum = (copy.deepcopy(fittest_chromosome.turbine),
                               copy.deepcopy(fittest_chromosome.get_fitness()))


    def _initialize_population(self, model_type, model_parameters):
        """
        Generate an initial random population of turbines within the turbine
        site limits
        """
        population = []
        for i in range(self._population_size):
            population.append(Chromosome(self.container, self._n_turbines))
        return population


    def get_fitnesses(self):
        """
        Return a sorted list of fitnesses with the chromosome indices
        """
        return self.fitnesses


    def update_fitnesses(self):
        """
        Updates the population fitness array
        """
        fitness = [self.population[i].get_fitness()
                   for i in range(self._population_size)]
        self.fitnesses = fitness


    def get_sorted_fitnesses(self):
        """
        Update the population fitness list from the stored chromosome values.
        Sort in fitness descending order with the index of the chromosome,
        i.e. [(41, f_max), .... , (22, f_min)]
        """
        fitness = enumerate(self.get_fitnesses())
        sorted_fitness = sorted(fitness, key=operator.itemgetter(1),
                                reverse=True)
        return sorted_fitness


# TODO: if crossover < 0.5; randomize remaining population
    def randomize(self):
        """
        Randomizes the remaining chromosomes
        """
        for i in range(self.n_crossover, self.population_size):
            seed = self.population_size*self.random_count + i
            self.population[i].randomize_m(seed)
        self.random_count += 1
