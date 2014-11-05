"""
Implements the Linear Superposition wake combination model.
"""
import numpy
from base_combination import WakeCombinationModel

class LinearSuperposition(WakeCombinationModel):
    """
    Implements the Linear Superposition wake combination model.
    """
    def __init__(self, flow_speed_at_turbine):
        super(LinearSuperposition, self).__init__(flow_speed_at_turbine)


    def reduce(self):
        """Linear Superposition.

        .. math:: \sum_j \left( 1 - \frac{u_{ij}}{u_j}\right)

        """
        flow_speeds = numpy.asarray(self.flow_speed_in_wake)
        reduced_speeds = flow_speeds/self.flow_speed_at_turbine
        return sum(1 - reduced_speeds)
