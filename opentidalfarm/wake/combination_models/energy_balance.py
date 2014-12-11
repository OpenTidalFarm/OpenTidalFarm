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
        """
        Combines a number of wakes to give a single flow speed at a turbine.

        See Renkema, D. [2007] section 4.8.1.
        """
        u_ij = numpy.asarray(self.u_ij)
        u_j = numpy.asarray(self.u_j)
        return numpy.sum((u_j**2-u_ij**2), axis=0)
