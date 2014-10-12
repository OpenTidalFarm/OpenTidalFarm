from .base_turbine import BaseTurbine
from .controls import Controls

class ThrustTurbine(BaseTurbine):
    def __init__(self, friction=12.0, diameter=20., minimum_distance=None,
                 controls=Controls(position=True)):

        # Check for a given minimum distance.
        if minimum_distance is None: minimum_distance=diameter*1.5
        # Initialize the base class.
        super(ThrustTurbine, self).__init__(friction=friction,
                                            diameter=diameter,
                                            minimum_distance=minimum_distance,
                                            controls=controls,
                                            thrust=True)
