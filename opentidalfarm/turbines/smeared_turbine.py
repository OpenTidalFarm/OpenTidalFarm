from .base_turbine import BaseTurbine
from .controls import Controls

class SmearedTurbine(BaseTurbine):
    def __init__(self, friction=12.0):

        # Initialize the base class.
        super(SmearedTurbine, self).__init__(friction=friction,
                                             controls=Controls(friction=True),
                                             smeared=True)
