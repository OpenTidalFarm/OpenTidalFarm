from generic_reduced_functional import GenericReducedFunctional                 

class ReducedFunctional(GenericReducedFunctional):
    """The master reduced functional that does the behind the scense management
    of the solver-specific reduced functionals
    """

    def __init__(self, farm, objective):
        """

        :param:
        :type:

        """
        self._farm = farm
        self._objective = objective
        self._controls = objective._controls
        m = self.farm.deserialize(self._controls)

 #       solvers = []
#        for objective in self._objective:
            #           solvers.append(objective.solver)
#
        #solvers = set(solvers)
        reduced_functionals = []
        if sw_solver in objective._solvers
            reduced_functionals.append(ShallowWaterReducedFunction(farm,
                self._objective.sw_objective)

        

        

    def reduced_functional(self, m):
        reduced_functional = 0




    def derivative(self, m_array):


    def hessian(self, m_array, m_dot_array):
