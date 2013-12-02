from opentidalfarm import AnalyticalWake
import numpy


class ChromosomeContainer(object):
    """
    Stores shard values used for each chromosome
    """
    def __init__(self, config, flow, limits, model_type, model_parameters):
        self._limits = limits
        self._turbine_radius = (config.params["turbine_x"] +
                                config.params["turbine_y"])/4.
        self.model = AnalyticalWake(config, flow, model_type, model_parameters)


class Chromosome(object):
    def __init__(self, container, n_turbines, turbines=None):
        self._container = container
        self._limits = self._container._limits
        self._turbine_radius = self._container._turbine_radius
        self._n_turbines = n_turbines
        self.model = self._container.model
        if turbines is None:
            self.turbine = self._initialize_chromosome()
        else:
            self.turbine = turbines
        self.current_fitness = None
        self.update_fitness()


    def __call__(self, turbines):
        """
        Update turbine on call
        """
        self.turbine = turbines
        self.update_fitness()


    def get_fitness(self):
        """
        Returns the current fitness value
        """
        return self.current_fitness


    def update_fitness(self):
        """
        Evaluates and sets the current fitness
        """
        self.current_fitness = self.model._total_power(self.turbine)


    def fitness(self):
        """
        Evaluates the fitness, sets it as the current value and then returns it
        """
        self.update_fitness()
        return self.current_fitness


    def _initialize_chromosome(self):
        """
        Randomize the turbines within the global limits
        """
        m = numpy.random.rand(self._n_turbines*2)
        lim = self._limits
        size = self._turbine_radius
        for i in range(len(m)):
            # limits are ordered [xmin, ymin, xmax, ymax] so use j to get the
            # right index for each i
            j = i%2
            m[i] = (lim[j+2] - lim[j] - size)*m[i] + lim[j] + 0.5*size
        return m


    def _randomize_chromosome(self):
        """
        Randomizes the chromosome solution within the limits, sets the new
        values and updates the fitness
        """
        m = self._initialize_chromosome()
        self.turbine = m
        self.update_fitness()


    def _within_limits(self):
        """
        Convenience method, returns whether the chromosomes set of turbines are
        fully within the limits of the domain
        """
        for i in range(len(self.turbine)):
            # x-coordinate
            if i%2==0:
                if (self.turbine[i] < self._limits[0] or
                    self.turbine[i] > self._limits[2]):
                    return False
            # y-coordinate
            else:
                if (self.turbine[i] < self._limits[1] or
                    self.turbine[i] > self._limits[3]):
                    return False
        # if we haven't been out of the limits then we're within the limits
        return True


class CopyChromosome(Chromosome):
    def __init__(self, chromosome, turbines):
        self._container = chromosome._container
        self._limits = self._container._limits
        self._turbine_radius = self._container._turbine_radius
        self._n_turbines = chromosome._n_turbines
        self.model = self._container.model
        self.turbine = turbines
        self.current_fitness = None
        self.update_fitness()

