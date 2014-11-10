"""
A WakeCombination class from which other wake combination models may be derived.
"""
import numpy

class WakeCombinationModel(object):
    """A base class from which wake combination models may be derived."""
    def __init__(self):
        # The speed at turbine i due to the wake of turbine j
        self.u_ij = []
        # The speed at turbine j (causing wake for turbine i)
        self.u_j = []


    def add(self, u_ij, u_j):
        """Adds speed_in_wake to the flow_speed_in_wake list."""
        try:
            assert isinstance(u_ij, (list, tuple, numpy.ndarray))
            assert isinstance(u_j, (list, tuple, numpy.ndarray))
            assert (len(u_ij)==2)
            assert (len(u_j)==2)
        except AssertionError:
            raise TypeError("'speed_in_wake' must be a list, tuple or "
                            "numpy.ndarray of length 2")
        self.u_ij.append(u_ij)
        self.u_j.append(u_j)


    def reduce(self):
        """Combines all the flow_speed_in_wake value into a single speed."""
        raise NotImplementedError("'combine' is not implemented in the base "
                                  "class")

    def _set_nan_or_inf_to_zero(self, array):
        array[numpy.isinf(array) + numpy.isnan(array)] = 0
        return array
