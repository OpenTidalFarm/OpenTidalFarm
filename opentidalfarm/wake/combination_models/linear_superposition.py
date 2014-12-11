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
        """
        Combines a number of wakes to give a single flow speed at a turbine.

        .. math:: \sum_j \left( 1 - \frac{u_{ij}}{u_j}\right)

        See Renkema, D. [2007] section 4.8.1.
        """
        u_ij = numpy.asarray(self.u_ij)
        u_j = numpy.asarray(self.u_j)

        u_ij = self._set_below_abs_tolerance_to_zero(u_ij)
        u_j = self._set_below_abs_tolerance_to_zero(u_j)

        with numpy.errstate(all='ignore'):
            result = u_ij/u_j
        result = self._set_nan_or_inf_to_zero(result)
        return numpy.sum((1 - result), axis=0)

