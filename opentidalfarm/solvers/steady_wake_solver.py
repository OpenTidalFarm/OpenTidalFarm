import numpy
from ..helpers import FrozenClass
from solver import Solver
from ..problems import SteadyWakeProblem


class SteadyWakeSolver(Solver):
    """A steady-state solver for Wake models."""
    def __init__(self, problem):
        if not isinstance(problem, SteadyWakeProblem):
            raise TypeError("'problem' must be of type SteadyWakeProblem")

        self.problem = problem
        self._farm = self.problem.parameters.tidal_farm


    def solve(self, turbines=None):
        """Returns an iterator for solving the steady wake problem."""

        flow_speeds = []
        if turbines is None:
            turbines = self._farm.turbine_positions
#        from IPython import embed; embed()
        wake_model = self.problem.parameters.wake_model
        for i, turbine_i in enumerate(turbines):
            combiner = self.problem.parameters.combination_model()
            other_turbines = numpy.delete(numpy.copy(turbines), i, axis=0)
            for turbine_j in other_turbines:
                u_i = wake_model.flow_at(turbine_i)
                flow_multiplier = wake_model.multiplier(turbine_i, turbine_j)
                u_j = wake_model.flow_at(turbine_j)
                u_ij = u_i*flow_multiplier
                combiner.add(u_ij, u_j)
            flow_speeds.append(u_i * combiner.reduce())

        #return numpy.asarray(flow_speeds)
        flow_speeds = numpy.asarray(flow_speeds)
#        from IPython import embed; embed()
        return flow_speeds
