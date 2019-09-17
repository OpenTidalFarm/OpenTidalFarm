from dolfin import *
from dolfin.cpp.log import log
from ufl import tanh
from .base_turbine import BaseTurbine
from .controls import Controls

class ThrustTurbine(BaseTurbine):
    """ Create a turbine that is modelled as a bump of bottom friction.
        In addition this turbine implements cut in and out speeds for the
        power production.

        This turbine introduces a non-linearity, which is handled explicitly. """
    def __init__(self,
                 friction=1.0,
                 diameter=20.,
                 swept_diameter=20.,
                 c_t_design=0.6,
                 cut_in_speed=1,
                 cut_out_speed=2.5,
                 water_depth = None,
                 upwind_correction=True,
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
        self.upwind_correction = upwind_correction

        # Check that the parameter choices make some sense - these won't break
        # the simulation but may give unexpected results if the choice isn't
        # understood.
        if self.swept_diameter != self.diameter:
            log(LogLevel.INFO, 'Warning - swept_diameter and plan_diameter are not equal')
        if self.friction != 1.0:
            log(LogLevel.INFO, 'Warning - for accurate parametrisation friction should \
                       be set to 1')

        swept_area = pi * (swept_diameter/2)**2
        plan_area = diameter**2
        # We can bundle all this up into a constant (i.e. independent of u) in
        # a stunning display of imaginative thinking we will call this the
        # turbine_parametrisation_constant
        self.turbine_parametrisation_constant = 0.5 * swept_area / \
                                    ((self._unit_bump_int/4) * plan_area)
        if upwind_correction:
            # This is a correction for the fact that C_t (c_t_design)
            # is defined as the thrust coefficient relative to the *upstream*
            # velocity. Since the drag term is computed from the local depth-averaged
            # velocity (which is lower) we need to compensate for this. The theory
            # behind this is explained in http://arxiv.org/abs/1506.03611

            if water_depth is None:
                raise ValueError("The water_depth needs to be specifed for the upwind correction")
            # the "numerical" cross-section this is the cross section over which
            # the drag is effectively applied
            effective_area = diameter*water_depth

            self.turbine_parametrisation_constant *= 4./(1.+sqrt(1-swept_area/effective_area*self.c_t_design))**2

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

