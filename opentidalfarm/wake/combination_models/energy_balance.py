"""
Implements the Energy Balance wake combination model.
"""
import numpy
from base_combination import WakeCombinationModel

class EnergyBalance(WakeCombinationModel):
    """
    Implements the Energy Balance wake combination model.
    """
    def __init__(self):
        super(EnergyBalance, self).__init__()


    def reduce(self):
        """Product of all flow speeds divided by the speed at the turbine."""
        at_turbine = numpy.asarray(self.flow_speed_in_wake)
        at_turbines_causing_wake = numpy.asarray(self.flow_speed_at_turbine)
        return numpy.sum((at_turbines_causing_wake**2-at_turbine**2), axis=0)
