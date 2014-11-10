"""
Implements the Geometric Sum wake combination model.
"""
import numpy
from base_combination import WakeCombinationModel

class GeometricSum(WakeCombinationModel):
    """
    Implements the Geometric Sum wake combination model.
    """
    def __init__(self):
        super(GeometricSum, self).__init__()


    def reduce(self):
        """Product of all flow speeds divided by the speed at the turbine."""
        u_ij = numpy.asarray(self.u_ij)
        u_j = numpy.asarray(self.u_j)
        # Set all results from of zero division to zero.
        with numpy.errstate(all='ignore'):
            result = u_ij/u_j
        result = self._set_nan_or_inf_to_zero(result)
        return numpy.prod(result, axis=0)
