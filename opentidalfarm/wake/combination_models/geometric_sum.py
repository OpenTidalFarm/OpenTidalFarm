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
        """
        Combines a number of wakes to give a single flow speed at a turbine.

        See Renkema, D. [2007] section 4.8.1.
        """
        u_ij = numpy.asarray(self.u_ij)
        u_j = numpy.asarray(self.u_j)
        # Set all results from of zero division to zero.
        with numpy.errstate(all='ignore'):
            result = u_ij/u_j
        result = self._set_nan_or_inf_to_zero(result)
        return numpy.prod(result, axis=0)
