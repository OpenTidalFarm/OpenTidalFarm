"""
Implements the Geometric Sum wake combination model.
"""
import numpy
from base_combination import WakeCombinationModel

class GeometricSum(WakeCombinationModel):
    """
    Implements the Geometric Sum wake combination model.
    """
    def __init__(self, flow_speed_at_turbine):
        super(GeometricSum, self).__init__(flow_speed_at_turbine)


    def reduce(self):
        """Product of all flow speeds divided by the speed at the turbine."""
        flow_speeds = numpy.asarray(self.flow_speed_in_wake)
        return flow_speeds.prod(axis=0)/self.flow_speed_at_turbine
