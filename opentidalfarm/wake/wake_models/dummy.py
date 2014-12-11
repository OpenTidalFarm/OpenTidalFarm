"""
Implements a dummy wake model for testing purposes. The model does not affect
the flow at all.
"""
from base_wake import WakeModel

class DummyWakeModel(WakeModel):
    """Implements a dummy wake model."""
    def __init__(self, flow_field):
        # Call the parent constructor.
        super(DummyWakeModel, self).__init__(flow_field)


    def in_wake(self, point_a, point_b):
        return True


    def multiplier(self, point_a, point_b):
        if self.in_wake(point_a, point_b):
            return 0.5
        else:
            return 1.0
