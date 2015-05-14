from .base_turbine import BaseTurbine
from .controls import Controls
from dolfin import *

class ThrustTurbine(BaseTurbine):
    def __init__(self,
                 friction=1.0,
                 diameter=20.,
                 swept_diameter=20.,
                 c_t_design=0.6,
                 cut_in_speed=1,
                 cut_out_speed=2.5,
                 minimum_distance=None,
                 controls=Controls(position=True)):

        # Check for a given minimum distance.
        if minimum_distance is None: minimum_distance=diameter*1.5
        # Initialize the base class.
        super(ThrustTurbine, self).__init__(friction=friction,
                                            diameter=diameter,
                                            minimum_distance=minimum_distance,
                                            controls=controls,
                                            thrust=True)

        # To parametrise a square 2D plan-view turbine to characterise a
        # realistic tidal turbine with a circular swept area in the section
        # plane we assume that the specified 2D turbine diameter is equal to the
        # circular swept diameter
        self.swept_diameter = swept_diameter
        self.c_t_design = c_t_design
        self.cut_in_speed = cut_in_speed
        self.cut_out_speed = cut_out_speed

        # Check that the parameter choices make some sense - these won't break
        # the simulation but may give unexpected results if the choice isn't
        # understood.
        if self.swept_diameter != self.diameter:
            log(INFO, 'Warning - swept_diameter and plan_diameter are not equal')
        if self.friction != 1.0:
            log(INFO, 'Warning - for accurate parametrisation friction should \
                       be set to 1')

        swept_area = pi * (swept_diameter/2)**2
        plan_area = diameter**2
        # We can bundle all this up into a constant (i.e. independent of u) in
        # a stunning display of imaginative thinking we will call this the
        # turbine_parametrisation_constant
        self.turbine_parametrisation_constant = swept_area / \
                                    ((self._unit_bump_int/4) * 0.5 * plan_area)

    def less_than_cut_out(self, u_mag):
        """ The function describing the thrust coefficient for velocities <
        cut_out_speed
        """
        return self.c_t_design * ((tanh(10*(u_mag-self.cut_in_speed))+1)/2)

    def greater_than_cut_out(self, u_mag):
        """ The function describing the thrust coefficient for velocities >
        cut_out_speed
        """
        return self.c_t_design * (self.cut_out_speed / u_mag)

    def compute_C_t(self, u_mag):
        """ Return C_t as a function of u_mag
        """
        return conditional(gt(u_mag, self.cut_out_speed),
                           self.greater_than_cut_out(u_mag),
                           self.less_than_cut_out(u_mag))

