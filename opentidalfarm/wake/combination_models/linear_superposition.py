"""
Implements the Linear Superposition wake combination model.
"""
import numpy
from base_combination import WakeCombinationModel

class LinearSuperposition(WakeCombinationModel):
    """
    Implements the Linear Superposition wake combination model.
    """
    def __init__(self):
        super(LinearSuperposition, self).__init__()


    def reduce(self):
        """Linear Superposition.

        .. math:: \sum_j \left( 1 - \frac{u_{ij}}{u_j}\right)

        """
        at_turbine = numpy.asarray(self.flow_speed_in_wake)
        at_turbines_causing_wake = numpy.asarray(self.flow_speed_at_turbine)
        with numpy.errstate(all='ignore'):
            result = at_turbine/at_turbines_causing_wake
        result = self._set_nan_or_inf_to_zero(result)
        return numpy.sum((1 - result), axis=0)

