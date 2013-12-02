import operator
import inspect
import numpy

class Selector(object):
    """
    Selector class for genetic algorithms -- each selection method returns the
    indices of the selected chromosomes within the population
    """
    def __init__(self, population, survival_rate, selection_type,
                 selection_options):
        # make a dictionary of class methods (key=method name)
        self.valid_types = dict(inspect.getmembers(self, inspect.ismethod))

        # check for valid selection_type
        if selection_type not in self.valid_types:
            raise KeyError("%s is not a valid selection_type" % selection_type)
        else:
            self.population = population
            self.selection_type = selection_type
            self.survival_rate = survival_rate
            proposed_size = self.population._population_size*self.survival_rate
            proposed_size = int(proposed_size)
            # ensure we have an even number of selected parents
            if proposed_size%2==0:
                self.proportion_size = proposed_size
            # else, thers an odd number and space to add one more
            elif proposed_size < (self.population._population_size-1):
                self.proportion_size = proposed_size + 1
            # else, theres an odd number and not space to add one
            else:
                self.proportion_size = proposed_size - 1
            self.selection_options = selection_options
            # set selection type to be the method corresponding to the type
            self.select = self.valid_types[self.selection_type]


    def tournament(self):
        """
        Tournament selection: selects from a random pool of chromosomes taken
        from the population. Selects chromosomes if a randomly chosen number is
        lower than the selection pressure. Continues to do so until the
        proportion size has been reached.
        """
        default_options = {"pool_proportion": 1, "selection_pressure": 1}
        if self.selection_options is None:
            self.selection_options = default_options
        else:
            for key in default_options:
                if key not in self.selection_options:
                    self.selection_options.update({key: default_options[key]})

        # generate a pool size based on the pool proportion
        pool_size = int(self.selection_options["pool_proportion"]*
                        self.population._population_size)
        if pool_size < self.proportion_size:
            raise ValueError("Pool size can't be smaller than the surving "
                            +"population size. Raise the pool proportion value")

        # pick the indices of the chromosomes to enter the pool
        indices = numpy.random.choice(range(self.population._population_size),
                                            pool_size, replace=False)

        # generate a list of tuples of index and fitness
        pool = []
        fitness = numpy.array(self.population.get_fitnesses())
        for i in indices:
            pool.append((i, fitness[i]))

        # sort the pool base on fitness (lowest to highest)
        sorted_fitness = sorted(pool, key=operator.itemgetter(1))

        selected = []
        while len(selected) < self.proportion_size:
            # select fittest
            if self.selection_options["selection_pressure"] > numpy.random.rand():
                # get index of the chromosome with highest fitness and remove
                # it from the pool
                selected.append(sorted_fitness[-1][0])
                sorted_fitness.pop(-1)
            # if not fittest -- get random chromosome
            else:
                chosen = numpy.random.choice(len(pool))
                selected.append(sorted_fitness[chosen][0])
                sorted_fitness.pop(chosen)

        # return indicies of selected parents
        return (numpy.array(selected)).astype(int)


    def roulette(self):
        """
        Routlette wheel selection: selects chromosomes in a fitness
        proportionate manner
        """
        sorted_fitness = numpy.array(self.population.get_sorted_fitnesses())

        def _accumulate(sorted_fitness):
            maximum = sorted_fitness[0][1]
            accumulated = [maximum]
            for i in range(1,len(sorted_fitness)):
                accumulated.append(accumulated[i-1]+sorted_fitness[i][1])
            accumulated = numpy.array(accumulated)/accumulated[-1]
            return accumulated

        selected = []
        for i in range(self.proportion_size):
            # generate cumulative fitness from a chromosome pool
            accumulated = _accumulate(sorted_fitness)
            # random value within between 0 and 1
            select = numpy.random.random()
            for j in range(len(accumulated)):
                if select < accumulated[j]:
                    selected.append(sorted_fitness[j][0])
                    sorted_fitness = numpy.delete(sorted_fitness, j, axis=0)
                    break
                else:
                    next

        # numpy doesn't allow tuples of floats and ints so converts ints to
        # float -- ensure that we return ints as they are used as indices
        return (numpy.array(selected)).astype(int)


    def best(self):
        """
        Take the best n chromosomes
        """
        sorted_fitness = self.population.get_sorted_fitnesses()
        selected = []
        for i in range(self.proportion_size):
            selected.append(sorted_fitness[i][0])
        # return list of selected chromosome indices
        return selected


    def stochastic_universal(self):
        """
        The stochastic universal sampling technique, see Baker 1987, "Reducing
        Bias and Inefficiency in the Selection Algorithm"
        """
        sorted_fitnesses = self.population.get_sorted_fitnesses()
        max_fitness = sorted_fitnesses[0][1]
        accumulated = [max_fitness]
        for i in xrange(1, len(sorted_fitnesses)):
            accumulated.append(accumulated[i-1]+sorted_fitnesses[i][1])
        accumulated = numpy.array(accumulated)/accumulated[-1]
        spacing = 1./self.proportion_size
        start = numpy.random.random()*spacing
        picking_points = [start+spacing*i for i in xrange(self.proportion_size)]
        picked = []
        for i in xrange(len(accumulated)):
            while (len(picked) < self.proportion_size and
                   picking_points[len(picked)] < accumulated[i]):
                picked.append(sorted_fitnesses[i][0])
            else:
                pass
        return picked
