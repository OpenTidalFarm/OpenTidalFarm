"""
Implements the Energy Balance wake combination model.
"""
import numpy
from base_combination import WakeCombinationModel

class SumOfSquares(WakeCombinationModel):
    """
    Implements the Energy Balance wake combination model.
    """
    def __init__(self):
        super(SumOfSquares, self).__init__()


    def reduce(self):
        """Product of all flow speeds divided by the speed at the turbine."""
        at_turbine = numpy.asarray(self.flow_speed_in_wake)
        at_turbines_causing_wake = numpy.asarray(self.flow_speed_at_turbine)
        with numpy.errstate(all="ignore"):
            result = at_turbine/at_turbines_causing_wake
        result = self._set_nan_or_inf_to_zero(result)
        return numpy.sum((1 - result)**2, axis=0)

