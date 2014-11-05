"""
A WakeCombination class from which other wake combination models may be derived.
"""
import numpy

class WakeCombinationModel(object):
    """A base class from which wake combination models may be derived."""
    def __init__(self, flow_speed_at_turbine):
        self.flow_speed_at_turbine = numpy.array(flow_speed_at_turbine)
        self.flow_speed_in_wake = []


    def add(self, speed_in_wake):
        """Adds speed_in_wake to the flow_speed_in_wake list."""
        try:
            assert isinstance(speed_in_wake, (list, tuple, numpy.ndarray))
            assert (len(speed_in_wake)==2)
        except AssertionError:
            raise TypeError("'speed_in_wake' must be a list, tuple or "
                            "numpy.ndarray of length 2")
        self.flow_speed_in_wake.append(speed_in_wake)


    def reduce(self):
        """Combines all the flow_speed_in_wake value into a single speed."""
        raise NotImplementedError("'combine' is not implemented in the base "
                                  "class")
