from .base_turbine import BaseTurbine
from .controls import Controls
from math import pi

class BumpTurbine(BaseTurbine):
    """ Create a turbine that is modelled as a bump of bottom friction """
    def __init__(self, thrust_coefficient=0.8, 
                 diameter=20., minimum_distance=None,
                 controls=Controls(position=True)):

        # Check for a given minimum distance.
        if minimum_distance is None: minimum_distance=diameter*1.5

        self.thrust_coefficient = thrust_coefficient
        swept_area = pi*(diameter/2.)**2

        # in the actual drag term we're using eqn (15) in [1], but the correction is only added in 
        # in the solver as it depends on the depth
        # [1] S.C. Kramer, M.D. Piggott http://doi.org/10.1016/j.renene.2016.02.022
        # so here c_t = friction * bump, friction = A_t*C_t/(2*integral(bump))
        friction = swept_area * thrust_coefficient / 2. / (self._unit_bump_int * diameter**2/4.)

        # Initialize the base class.
        super(BumpTurbine, self).__init__(friction=friction,
                                          diameter=diameter,
                                          minimum_distance=minimum_distance,
                                          controls=controls,
                                          bump=True)
