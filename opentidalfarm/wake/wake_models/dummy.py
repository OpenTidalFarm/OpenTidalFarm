"""
Implements a dummy wake model for testing purposes. The model does not affect
the flow at all.
"""
from base_wake import WakeModel
import numpy

class DummyWakeModel(WakeModel):
    """Implements a dummy wake model."""
    def __init__(self, flow_field):
        # Call the parent constructor.
        super(DummyWakeModel, self).__init__(flow_field)


    def in_wake(self, turbine, another_turbine):
        """
        True if turbine is in the wake of another_turbine.
        """
        return True


    def multiplier(self, turbine, another_turbine):
        if self.in_wake(turbine, another_turbine):
            return numpy.random.rand()
        else:
            return 1.0
