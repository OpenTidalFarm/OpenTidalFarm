from chromosome import CopyChromosome
import inspect
import numpy

class Crossover(object):
    """
    A crossover class. Once initialised, call self.crossover() to return new
    children
    """
    def __init__(self, population, crossover_type, survival_rate,
                 selection_type, selection_options):
        # make a dictionary of class methods (key=method name)
        self.valid_types = dict(inspect.getmembers(self, inspect.ismethod))

        # check for valid crossover types
        if crossover_type not in self.valid_types:
            raise KeyError("%s is not a valid crossover type" % selection_type)
        else:
            self.crossover_type = crossover_type
            self.population = population
            self.mate = self.valid_types[self.crossover_type]


    def _get_n_children(self, n_parents):
        """
        Returns the number of children to generate
        """
        # if parents make up half or more than half the population
        if n_parents >= (self.population._population_size/2):
            return self.population._population_size - n_parents
        # parents make up less than half
        else:
            return n_parents


    def one_point(self, parents):
        """
        One point crossover -- splices two parents at the same index and swaps
        the trailing part of the chromosome
        """
        number_of_children = self._get_n_children(len(parents))
        # generate a list of parent index couples to produce offspring
        couples = numpy.random.choice(parents,
                                      (max(number_of_children/2, 1), 2),
                                      replace=False)
        size = self.population._n_turbines*2
        children = []
        for couple in couples:
            # select a crossover index
            point = numpy.random.choice(size)
            # if end values, then there's no need to crossover
            if point==0 or point==size:
                children.append(self.population.population[couple[0]].turbine)
                children.append(self.population.population[couple[1]].turbine)
            else:
                # split the parents turbine list at point
                child_a1 = self.population.population[couple[0]].turbine[:point]
                child_a2 = self.population.population[couple[0]].turbine[point:]
                child_b1 = self.population.population[couple[1]].turbine[:point]
                child_b2 = self.population.population[couple[1]].turbine[point:]
                child_a = numpy.concatenate((child_a1, child_b2))
                child_b = numpy.concatenate((child_b1, child_a2))
                children.append(child_a)
                children.append(child_b)
        if len(children) > number_of_children:
            children = children[:number_of_children]
        return children


    def two_point(self, parents):
        """
        Two point crossover -- splices two parents twice at the same indices and
        swaps the middle section
        """
        number_of_children = self._get_n_children(len(parents))
        # generate a list of couples to produce offspring
        couples = numpy.random.choice(parents,
                                      (max(number_of_children/2, 1), 2),
                                      replace=False)
        size = self.population._n_turbines*2
        children = []
        for couple in couples:
            # select a crossover index
            points = numpy.random.choice(size, 2, replace=False)
            # if end values, then there's no need to crossover
            if ((points[0]==0 and points[1]==size) or
                (points[1]==0 and points[0]==size)):
                children.append(self.population.population[couple[0]].turbine)
                children.append(self.population.population[couple[1]].turbine)
            else:
                # sort the crossing points
                points.sort()
                # get copies of the parent turbines
                parent_a = numpy.array(self.population.population[couple[0]].turbine)
                parent_b = numpy.array(self.population.population[couple[1]].turbine)

                child_a1 = parent_a[:points[0]]
                tmp = parent_a[points[0]:]
                child_a2 = tmp[:points[1]]
                child_a3 = tmp[points[1]:]

                child_b1 = parent_b[:points[0]]
                tmp = parent_b[points[0]:]
                child_b2 = tmp[:points[1]]
                child_b3 = tmp[points[1]:]

                child_a = numpy.concatenate((child_a1, child_b2, child_a3))
                child_b = numpy.concatenate((child_b1, child_a2, child_b3))

                children.append(child_a)
                children.append(child_b)
        if len(children) > number_of_children:
            children = children[:number_of_children]
        return children


    def uniform(self, parents):
        """
        Evaluates a each bit in the chromosome and crosses over with a
        probability of 0.5. More exploratory than one_point and two_point
        """
        number_of_children = self._get_n_children(len(parents))
        # generate a list of couples to produce offspring
        couples = numpy.random.choice(parents,
                                      (max(number_of_children/2, 1), 2),
                                      replace=False)
        size = self.population._n_turbines*2
        children = []
        for couple in couples:
            parent_a = numpy.array(self.population.population[couple[0]].turbine)
            parent_b = numpy.array(self.population.population[couple[1]].turbine)
            # generate a list of random integers between 0 and 1
            to_cross = numpy.random.randint(0, 2, size)
            for i in range(len(to_cross)):
                # swap a bit
                if to_cross[i] > 0:
                    # in place swapping of coordinates
                    parent_a[i], parent_b[i] = parent_b[i], parent_a[i]
                else:
                    pass
            children.append(parent_a)
            children.append(parent_b)
        if len(children) > number_of_children:
            children = children[:number_of_children]
        return children


    def blending(self, parents):
        """
        Chromosome blending (see Wright 1991), create three children and then
        kill the weakest before passing the fittest two to the next generation
        """
        number_of_children = self._get_n_children(len(parents))
        # generate a list of couples to produce offspring (ensure at least one
        # couple)
        couples = numpy.random.choice(parents,
                                      (max(number_of_children/2, 1), 2),
                                      replace=False)
        size = self.population._n_turbines*2
        children = []
        for couple in couples:
            child_1 = []
            child_2 = []
            child_3 = []
            parent_1 = numpy.array(self.population.population[couple[0]].turbine)
            parent_2 = numpy.array(self.population.population[couple[1]].turbine)
            for i in range(size):
                child_1.append(0.5*(parent_1[i]) + 0.5*(parent_2[i]))
                child_2.append(1.5*(parent_1[i]) - 0.5*(parent_2[i]))
                child_3.append(1.5*(parent_2[i]) - 0.5*(parent_1[i]))

            temp_a = CopyChromosome(self.population.population[couple[0]], child_1)
            temp_b = CopyChromosome(self.population.population[couple[0]], child_2)
            temp_c = CopyChromosome(self.population.population[couple[0]], child_3)
            proposed_children = [temp_a, temp_b, temp_c]
            within_limits = [child._within_limits() for
                             child in proposed_children]
            in_limits = within_limits.count(True)
            # best case; one turbine out of limits, take other two turbines
            if in_limits==2:
                proposed_children.pop(within_limits.index(False))
                children.append(proposed_children[0].turbine)
                children.append(proposed_children[1].turbine)
            # one turbine within limits, so randomize another turbine
            elif in_limits==1:
                children.append((proposed_children.pop(within_limits.index(True))).turbine)
                children.append(proposed_children[0]._initialize_chromosome())
            # no turbines within limits, randomize two turbines
            elif in_limits==0:
                children.append(proposed_children[0]._initialize_chromosome())
                children.append(proposed_children[1]._initialize_chromosome())
            # all in limits, take fittest two
            else:
                fitneses = [child.get_fitness() for child in proposed_children]
                index = fitneses.index(min(fitneses))
                proposed_children.pop(index)
                children.append(proposed_children[0].turbine)
                children.append(proposed_children[1].turbine)

        # in case we have too many children
        if len(children) > number_of_children:
            children = children[:number_of_children]
        return children


    def blend_radcliffe(self, parents):
        """
        Chromosome blending (see Radcliffe 1991)
        """
        number_of_children = self._get_n_children(len(parents))
        # generate a list of couples to produce offspring
        couples = numpy.random.choice(parents,
                                      (max(number_of_children/2, 1), 2),
                                      replace=False)
        children = []
        for c in couples:
            child_1 = []
            child_2 = []
            parent_1 = self.population.population[c[0]].turbine
            parent_2 = self.population.population[c[1]].turbine
            beta = numpy.random.random()
            for i in range(self.population._n_turbines*2):
                child_1.append(beta*parent_1[i] + (1-beta)*parent_2[i])
                child_2.append((1-beta)*parent_1[i] + beta*parent_2[i])
            children.append(child_1)
            children.append(child_2)

        # in case we have too many children
        if len(children) > number_of_children:
            children = children[:number_of_children]
        return children


    def haupt(self, parents):
        """
        Chromosome blending as per Haupt 2004
        """
        number_of_children = self._get_n_children(len(parents))
        # generate a list of couples to produce offspring
        couples = numpy.random.choice(parents,
                                      (max(number_of_children/2, 1), 2),
                                      replace=False)
        children = []
        for c in couples:
            crossing_point = numpy.random.randint(self.population._n_turbines*2)
            beta = numpy.random.random()
            child_a = []
            child_b = []
            parent_a = self.population.population[c[0]].turbine
            parent_b = self.population.population[c[1]].turbine
            for i in range(crossing_point):
                child_a.append(parent_a[i])
                child_b.append(parent_b[i])
            for i in range(crossing_point, self.population._n_turbines*2):
                child_a.append(parent_a[i] - beta*(parent_a[i] - parent_b[i]))
                child_b.append(parent_b[i] - beta*(parent_b[i] - parent_a[i]))
            children.append(child_a)
            children.append(child_b)

        # in case we have too many children
        if len(children) > number_of_children:
            children = children[:number_of_children]
        return children
