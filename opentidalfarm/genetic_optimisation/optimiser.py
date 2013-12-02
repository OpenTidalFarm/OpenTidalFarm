from dolfin import info, info_red, info_blue, info_green
from population import Population
from generate import Generate
import numpy
import time
import pylab
import csv

class GeneticOptimisation(object):
    """
    Genetic optimisation for wake models.
    """

    def __init__(self,
                 config,
                 population_size,
                 maximum_iterations,
                 number_of_turbines,
                 ambient_flow,
                 wake_model_type="ApproximateShallowWater",
                 wake_model_parameters=None,
                 survival_rate=0.7,
                 crossover_type="uniform",
                 mutation_type="fitness_proportionate",
                 selection_type="best",
                 mutation_probability=0.07,
                 selection_options=None,
                 options=None):
        """
        Initializes the GeneticOptimisation problem

        Args:
            config: an OpenTidalFarm configuration
            population_size: the number of solutions in the population
            maximum_iterations: maximum number of iterations (generations)
            number_of_turbines: number of turbines to optimise
            ambient_flow: the background flow field (can be generated in
                OpenTidalFarm with a call to helpers.get_ambient_flow(config))
            wake_model_type: with the name of the wake model to use
            wake_model_parameters: a dictionary ofparameters to use with the
                wake model, see individual models for more info on the
                parameters assoicated with each
            survival_rate: the proportion of the population that will be kept
                (and potentially mutated) in the next generation
            crossover_type: string, type of crossover to use
            mutation_type: string, type of mutation to use
            selection_type: string, type of selection to use
            mutation_probability: float, probability of a mutation (with
                fitness_proportionate mutation this is the maximum probability
                of mutation)
            selection_options: any options associated with the selection type
            options: dict, any other options, descritpions given below:
                default = {"jump_start": <False> - reinitialize a population but
                                with a chromosome that has the solution of the
                                fittest solution in the previous population. If
                                given an <int>, the population will be jump
                                started this many times
                           "initial_population_seed": <None> - a list of turbine
                                position vectors to see the initial population
                                with
                           "disp": <True> - print stats during the optimisation
                           "disp_normalized": <False> - normalize stats to the
                                mean fitness of the first population
                           "update_every": <int=10> - print stats every int
                                iterations
                           "predict_time": <True> - predict time until the
                                iteration limit is exceeded
                           "save_stats": <True> - save statistice to a file
                           "save_fitness": <False> - save the fitness values of
                                each solution to a file
                           "save_every": <int=1> - save stats every int
                                iterations
                           "stats_file": <string="optimisation_stats.csv"> -
                                save stats to string
                           "fitness_file": <string="optimisation_fitness.csv"> -
                                save fitness values to string
                           "plot": <True> - save a plot of the statistics
                           "plot_file": <string="population_fitness.png"> - save
                                the plot to string}

        """
        default_options = {"jump_start": False,
                           "initial_population_seed": None,
                           "disp": True,
                           "disp_normalized": False,
                           "update_every": 10,
                           "predict_time": True,
                           "save_stats": True,
                           "save_fitness": False,
                           "save_every": 1,
                           "stats_file": "optimisation_stats.csv",
                           "fitness_file": "optimisation_fitness.csv",
                           "plot": True,
                           "plot_file": "population_fitness.png"
                          }

        ### default_options:
        # jump_start: reinitialize a population but with a chromosome that has
        #             fittest solution from the previous population, set to an
        #             int for the number of times to jump start
        # initial_population_seed: a list of flattened arrays of turbine
        #                          positions to seed the initial population
        #                          with, if there are fewer seeds than the given
        #                          population size then the remaining
        #                          chromosomes will be randomized
        # disp: print info
        # disp_normalized: normalize the power and standard deviation of the
        #                  output to the average power of the initial population
        # update_every: print info every <iterations>
        # predict_time: print a prediction of the time left before iteration
        #               limit
        # save_stats: write stats to a file
        # save_fitness: save the fitness of each chromosome to file
        # save_every: save info every <iterations>
        # stats_file: where to save stats
        # fitness_file: where to save fitness
        # plot: plot stats upon optimisation completion
        # plot_file: save plot to file

        # check if options are given
        if options is None:
            self.options = default_options
        # check against the default
        else:
            for key in default_options:
                if key not in options:
                    options.update({key: default_options[key]})
            self.options = options

        # termination info
        self.iterations = 0
        self.maximum_iterations = maximum_iterations

        # for timing prediction
        self._start_time = None

        # initialize a population -- ensure we don't calculate gradients as we
        # dont need them
        if wake_model_parameters is not None:
            wake_model_parameters.update({"compute_gradient": False})
        else:
            wake_model_parameters = {"compute_gradient": False}
        self.population = Population(config, population_size,
                                     number_of_turbines, ambient_flow,
                                     wake_model_type, wake_model_parameters,
                                     self.options["initial_population_seed"])
        # initialize a generator
        self.generator = Generate(self.population, survival_rate,
                                  crossover_type, mutation_type, selection_type,
                                  mutation_probability, selection_options)


    def _print_info(self):
        """
        Print some information about the set up
        """
        info_blue(" Set up information ".center(80, "-"))
        info("  Population size: %i" % self.population._population_size)
        info("  Number of turbines: %i" % self.population._n_turbines)
        info("  Survival rate: %.2f" % self.generator.selector.survival_rate)
        info("  Crossover type: %s" % self.generator.crossover.crossover_type)
        info("  Selection type: %s" % self.generator.selector.selection_type)
        info("  Mutation type: %s" % self.generator.mutator.mutation_type)
        info("  Mutation probability: %.2f" % self.generator.mutator.mutation_probability)
        if self.options["jump_start"]:
            info("  Number of jump starts: %i" % self.options["jump_start"])
        else:
            info("  Jump start: False")


    def _print_options(self):
        """
        Print some set up options
        """
        info_blue(" Set up options ".center(80, "-"))
        info("  Display info: %s" % self.options["disp"])
        if self.options["disp"]:
            info("  Info interval: %i" % self.options["update_every"])
            info("  Predict remaining time: %s" % self.options["predict_time"])
        info("  Save stats: %s" % self.options["save_stats"])
        if self.options["save_stats"]:
            info("  Save every: %i iterations" % self.options["save_every"])
            info("  Save statistics to: %s" % self.options["stats_file"])
            info("  Plot statistics at end: %s" % self.options["plot"])
            if self.options["plot"]:
                info("  Save plot as %s" % self.options["plot_file"])


    def _get_average_fitness(self):
        """
        Returns the average fitness of the population for the current generation
        """
        total_fitness = sum(self.population.fitnesses)
        return total_fitness/float(self.population._population_size)


    def _get_end_fitnesses(self):
        """
        Returns the maximum and minimum fitnesses of the current generation
        """
        return max(self.population.fitnesses),\
               min(self.population.fitnesses)


    def _get_stats(self):
        current_max, current_min = self._get_end_fitnesses()
        current_average = self._get_average_fitness()
        global_max = self.population.global_maximum[1]
        std = numpy.std(self.population.get_fitnesses())
        return {"max": current_max,
                "min": current_min,
                "avg": current_average,
                "best": global_max,
                "std": std
               }


    def _print_stats(self, iteration):
        """
        Prints stats about the current state
        """
        stats = self._get_stats()
        str_iter = ("| %8i" % (iteration))
        str_power = (" | %16.2f / %16.2f / %16.2f (Min/Mean/Max) " %
                     (stats["min"]/self._normalize_fac,
                      stats["avg"]/self._normalize_fac,
                      stats["max"]/self._normalize_fac))
        str_global = ("| Global Max: %16.2f" %
                     (stats["best"]/self._normalize_fac))
        str_std = (" | Std: %16.2f" % (stats["std"]/self._normalize_fac))
        print_string = str_iter + str_power + str_global + str_std
        if self.options["predict_time"] and iteration > 0:
            elapsed = time.time() - self._start_time
            # as zero based, add 1
            hits = (iteration+1)/self.options["update_every"]
            dt = elapsed/hits
            seconds = int(((self.maximum_iterations-iteration)/
                            self.options["update_every"])*dt)
            mins, secs  = divmod(seconds, 60)
            hours, mins = divmod(mins, 60)
            days, hours = divmod(hours, 24)
            str_time = (" | %id %2ih %2im %2is remaining"
                        % (days, hours, mins, secs))
            print_string += str_time
        # print some stats
        info(print_string)


    def _save_header(self):
        """
        Writes a header to the save file
        """
        with open(self.options["stats_file"], "w") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(["Iteration", "Minimum", "Mean", "Maximum",
                                "GlobalMaximum", "StandardDeviation", "Time"])

        with open(self.options["fitness_file"], "w") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(["ChromosomeFitness"])


    def _save_stats(self, iteration):
        """
        Saves output to file
        """
        stats = self._get_stats()
        with open(self.options["stats_file"], "a") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow([iteration, stats["min"], stats["avg"],
                                stats["max"], stats["best"], stats["std"],
                                time.time()])

        fitness_data = self.population.get_fitnesses()
        with open(self.options["fitness_file"], "a") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(fitness_data)


    def _plot_data(self):
        """
        Reads data from the ouputted csv and plots it
        """
        iterations = []
        iter_min = []
        iter_mean = []
        iter_max = []
        global_max = []
        iter_std = []
        stats = []

        with open(self.options["stats_file"], "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            # skip the headers
            csvreader.next()
            for row in csvreader:
                iterations.append(int(row[0]))
                iter_min.append(float(row[1]))
                iter_mean.append(float(row[2]))
                iter_max.append(float(row[3]))
                global_max.append(float(row[4]))
                iter_std.append(float(row[5]))

        with open(self.options["fitness_file"], "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            csvreader.next()
            for row in csvreader:
                ind = []
                for subrow in row:
                    ind.append(float(subrow))
                stats.append(ind)

        # normalize data
        iter_min = numpy.array(iter_min)
        iter_mean = numpy.array(iter_mean)
        iter_max = numpy.array(iter_max)
        iter_std = numpy.array(iter_std)
        global_max = numpy.array(global_max)
        power_norm = iter_mean[0]
        iter_min /= power_norm
        iter_mean /= power_norm
        iter_max /= power_norm
        global_max /= power_norm
        stats /= power_norm

        green='#859900'
        blue='#268bd2'
        orange='#cb4b16'
        base='#002b36'
        pylab.fill_between(iterations, iter_max, iter_min, alpha=0.1, color=blue)
        _mean       = pylab.plot(iterations, iter_mean, linestyle='-', color=blue)
        _fitness    = pylab.plot(iterations, stats, linestyle='None', color=base, marker='.', markersize=0.2)
        _itermin    = pylab.plot(iterations, iter_min, linestyle='-', color=orange)
        _global_max = pylab.plot(iterations, global_max, linestyle='-', color=green)
        pylab.legend([_global_max[0], _mean[0], _itermin[0]],
                     ["Global Maximum", "Mean", "Minimum"],
                     loc=4, prop={"size":10})
        pylab.title("Population Fitness Statistics")
        pylab.xlabel("Generation (#)")
        pylab.ylabel("Normalised Fitness")
        pylab.grid()
        pylab.savefig(self.options["plot_file"])


    def exit_criteria_met(self):
        """
        Exit criteria
        """
        stats = self._get_stats()
        has_converged = ((stats["max"]-stats["min"]) < 1e-4)
        iteration_limit_reached = (self.iterations > self.maximum_iterations)
        return has_converged or iteration_limit_reached


    def run(self):
        """
        Run the genetic algorithm.
        """
        start_message = (" Starting the optimisation problem ").center(80, "-")
        warning_message = ("(this could take a while so you might want to get "
                          +"a cup of tea of something)").center(80)
        # print some options
        self._print_info()
        self._print_options()

        # write a header to the save file
        if self.options["save_stats"]:
            self._save_header()

        # print the starting messages
        info_red(start_message)
        info(warning_message)
        info_red("-"*80)

        # run the optimisation
        self._start_time = time.time()
        self._optimise()
        # if we "jump_start" we reinitialize the population with one chromosome
        # that contains the fittest solution from the previous population
        if self.options["jump_start"]:
            for i in range(self.options["jump_start"]):
                self.iterations += 1
                info_blue("Reinitializing the population...")
                # get best solution turbines
                best_solution = numpy.array(self.population.global_maximum[0])
                # set the other seed solutions
                if len(self.population.seeds)-1 < self.population._population_size:
                    for i in range(len(self.population.seeds)-1):
                        self.population.population[i](self.population.seeds[i])
                    for i in range(len(self.population.seeds)-1,
                                   self.population._population_size):
                        self.population.population[i]._randomize_chromosome()
                # randomize all chromosomes apart from the first
                else:
                    for i in range(len(self.population.population)-1):
                        self.population.population[i+1]._randomize_chromosome()
                # set the first chromosome to the best solution (also updates
                # fitness)
                self.population.population[0](best_solution)
                # update the population fitnesses
                self.population.update_fitnesses()
                self._optimise()
        # end the timer
        elapsed_time = time.time() - self._start_time

        # finish up
        end_message = (" Optimisation complete ").center(80, "-")
        info_green(end_message)
        info_green(" %i iterations took %.2f seconds ".center(80, "-")
             % (self.iterations, elapsed_time))
        solution_variables, solution_fitness = self.population.global_maximum
        info_blue("Solution fitness: %.2f" % solution_fitness)

        if self.options["save_stats"]:
            info("Optimisation statistics saved to %s" % self.options["stats_file"])
            if self.options["plot"]:
                self._plot_data()
                info("Fitness data plotted and saved to %s" % self.options["plot_file"])

        self.final_turbines = solution_variables


    def _optimise(self):
        """
        Performs the actual optimization
        """
        if self.options["disp_normalized"]:
            self._normalize_fac = self._get_stats()["avg"]
        else:
            self._normalize_fac = 1.

        # check for options before running to minimise checking if statements
        if self.options["disp"] and self.options["save_stats"]:
            while not self.exit_criteria_met():
                if self.iterations%self.options["update_every"]==0:
                    self._print_stats(self.iterations)
                if self.iterations%self.options["save_every"]==0:
                    self._save_stats(self.iterations)
                self.generator.generate()
                self.iterations += 1
            # save and print final stats
            self._print_stats(self.iterations)
            self._save_stats(self.iterations)

        # only print updates
        elif self.options["disp"]:
            while not self.exit_criteria_met():
                if self.iterations%self.options["update_every"]==0:
                    self._print_stats(self.iterations)
                self.generator.generate()
                self.iterations += 1
            # print final stats
            self._print_stats(self.iterations)

        # only save data
        elif self.options["save_stats"]:
            while not self.exit_criteria_met():
                if self.iterations%self.options["save_every"]==0:
                    self._save_stats(self.iterations)
                self.generator.generate()
                self.iterations += 1
            # save final step
            self._save_stats(self.iterations)

        # neither print not save data
        else:
            while not self.exit_criteria_met():
                self.generator.generate()


    def get_turbine_pos(self):
        """
        Returns the optimum turbine positions
        """
        try:
            assert(self.final_turbines is not None)
        except:
            raise RuntimeError("The population hasn't been optimised yet!")
        return self.final_turbines
