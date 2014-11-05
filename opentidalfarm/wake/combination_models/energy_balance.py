"""
Implements the Energy Balance wake combination model.
"""
import numpy
from base_combination import WakeCombinationModel

class EnergyBalance(WakeCombinationModel):
    """
    Implements the Energy Balance wake combination model.
    """
    def __init__(self, flow_speed_at_turbine):
        super(EnergyBalance, self).__init__(flow_speed_at_turbine)


    def reduce(self):
        """Product of all flow speeds divided by the speed at the turbine."""
        flow_speeds = numpy.array(self.flow_speed_in_wake)
        return numpy.sum((self.flow_speed_at_turbine**2-flow_speeds**2), axis=0)
