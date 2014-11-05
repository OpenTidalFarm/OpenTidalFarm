"""
Implements the Energy Balance wake combination model.
"""
import numpy
from base_combination import WakeCombinationModel

class SumOfSquares(WakeCombinationModel):
    """
    Implements the Energy Balance wake combination model.
    """
    def __init__(self, flow_speed_at_turbine):
        super(SumOfSquares, self).__init__(flow_speed_at_turbine)


    def reduce(self):
        """Product of all flow speeds divided by the speed at the turbine."""
        flow_speeds = numpy.asarray(self.flow_speed_in_wake)
        reduced_speeds = flow_speeds/self.flow_speed_at_turbine
        return sum((1 - reduced_speeds)**2)

