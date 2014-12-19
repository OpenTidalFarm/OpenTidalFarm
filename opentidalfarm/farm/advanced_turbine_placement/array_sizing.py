import numpy
import dolfin
import opentidalfarm as otf
from sklearn.gaussian_process import GaussianProcess
from hybrid_greedy_turbine_placement import HybridGreedyTurbinePlacement

# from operator import attrgetter
# list.sort(key=attrgetter('attribute'), reverse=True)
#
#class Sample(object):
#    """ A storage vessel to contain the details of each sample run
#    """
#    
#    def __init__(self, number_of_turbines, functional, power, fidelity,
#                 turbine_locations):
#
#        self.size = number_of_turbines
#        self.functional = functional
#        self.power = power
#        self.fidelity = fidelity
#        self.turbine_locations = turbine_locations


class ArraySizer(object):
    """ Class to determine the optimum size of the turbine array
    """

    def __init__(self, reduced_functional):
        """ Constructor method
        """

        # Check the reduced_functional is what we want it to be
        if not isinstance(reduced_functional,
                otf.reduced_functional_prototype.ReducedFunctionalPrototype):
            raise TypeError, "reduced_functional must be of type ReducedFunctional"

        # Fish out some useful objects for ease of accessibility
        self.rf = reduced_functional
        self.solver = reduced_functional.solver
        self.problem = self.solver.problem
        self.problem_params = reduced_functional._problem_params
        self.farm = reduced_functional._problem_params.tidal_farm 
        self.functional = reduced_functional.functional

        # We're going to want to keep some records of what we've tested
        self.record = self.farm.turbine_placement_parameters.record


    def initialise_population(self):
        """ Use the hybrid greedy advanced turbine layout technique to
        initialise our search.
        """
        self.farm.add_advanced_turbine_layout()

        initialise = HybridGreedyTurbinePlacement(self.rf)
        initialise.place()
        dolfin.info('Initialisation complete...')


    def generate_gpr(self):
        """ Takes the sampled points and generates the input/output map
        """
        # Extract the data we want to plot
        data = {}
        for i in self.record:
            data.append(i.size, i.functional)
        
        # Instantiate a Gaussian Process model
        gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, 
                             thetaU=1e-1, random_start=100)



    def size(self):
        """ Runs the array-sizing show
        """
        self.initialise_population()


# based on the improvement from placed to optimised (after initialising the
# population) we can scale the low fidelity samples and construct the first GPR.
