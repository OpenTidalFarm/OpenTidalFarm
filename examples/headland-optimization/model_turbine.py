from math import pi

class ModelTurbine(object):

    def __init__(self,
            blade_radius=(60/0.86)**0.5,   # Originally we had a turbine
            # diameter of 10 m and a Ct value of 0.6, but really we wanted ct
            # value of 0.86. Hence this conversion factor so that all model
            # results are still valid (it turns out that this blade radious is
            # more realistic anyway)
            minimum_distance=40,
            water_density=1e3,
            Ct=0.86,
            ):

        # Radius of turbine blades in m
        self.blade_radius = blade_radius

        # Minimum distance between two turbines in m
        self.minimum_distance = minimum_distance

        # Water density in kg/m^3
        self.water_density = water_density

        # Turbine thrust coefficient
        self.Ct = Ct

    @property
    def blade_diameter(self):
        ''' Returns the turbine diameter. '''
        return 2 * self.blade_radius

    @property
    def turbine_cross_section(self):
        ''' Returns the area that the turbine blades cover. '''
        return pi * self.blade_radius**2

    @property
    def maximum_density(self):
        ''' Returns the maximum turbine density
        '''
        dt_max = 1./self.minimum_distance**2
        return dt_max

    @property
    def maximum_smeared_friction(self):
        ''' Returns the average friction of a densely packed turbine area.
        '''
        return self.friction * self.maximum_density

    @property
    def bump_height(self, apply_roc_correction=False):
        ''' Returns the peak value of OpenTidalFarm's turbine representation.
            Note: The amount of friction in the smeared and discrete representation
            are not the same. Instead the computation is based on setting the applied
            force on the flow equal.

            This is based on:
            F_turbine = 0.5 * rho * C_t * A_t u_upstream**2                           # Thrust force of a single turbine
                      = 0.5 * rho * C_T * 4 / (1 + sqrt(1-C_T))**2 A_T u_turbine**2   # Apply Rocs correction
                      = rho int bump_function(x) c_t u**2                             # Set the force equal to the force of a discrete numpy turbine
                      is approximately
                        rho int bump_function(x) c_t u_turbine**2                     # Assume that u and u_turbine are the same

           =>
           c_t = 4/(1+sqrt(1-C_T))**2 C_T A_T / (2 \int bump_function(x))
        '''

        roc_correction_factor = 4./(1+(1-self.Ct)**0.5)**2
        A_T = self.turbine_cross_section
        int_bump_function = 1.45661 * self.blade_radius**2

        c_t = self.Ct * A_T / (2. * int_bump_function)
        if apply_roc_correction:
            c_t *= roc_correction_factor

        return c_t

    @property
    def cost_coefficient(self, average_power_factor=1, peak_velocity=2.0, profit_margin=0.4):
        """ Returns the cost coefficient for OpenTidalFarm.

            Parameters:
            - average_power_factor (%) specifies which percentage of the power
              production of the turbine in average over the power production at peak flow.
              The value 1. assumes a constant flow.
            - peak_velocity specifies the peak velocity of the sinusoidal flow.
            - profit_margin specifies the profit margin of the turbine over its
              entire life cycle.
        """

        c_slL = average_power_factor * (1-profit_margin) * self.water_density * peak_velocity**3

        return c_slL

    def number_of_turbines(self, friction_integral):
        return friction_integral / self.friction


    @property
    def friction(self):
        return self.Ct * self.turbine_cross_section / 2.


    def __str__(self):
        s =  """Model turbine specification
---------------------------
    Blade diameter                                   {} m.
    Minimum distance between two turbines:           {} m.
    Turbine cross section (A_T)                      {} m^2.
    Turbine induced friction:                        {}.
    Average turbine friction of densely packed farm: {}.
    Maximum turbine density:                         {}.
    OpenTidalFarm bump maximum:                      {}.
    OpenTidalFarm cost coefficient:                  {}.
        """.format(self.blade_diameter,
                   self.minimum_distance,
                   self.turbine_cross_section,
                   self.friction,
                   self.maximum_smeared_friction,
                   self.maximum_density,
                   self.bump_height,
                   self.cost_coefficient)
        return s


if __name__ == "__main__":
    model = ModelTurbine()
    print model
