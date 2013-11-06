from model import Model
import numpy

class Larsen(Model):
    """
    Defines the Larsen wake model
    """
    def __init__(self, flow_field, turbine_radius, model_parameters=None):
        def _setup(parameters):
            """
            Calculates model parameters used in the Larsen model and returns a
            dictionary of the extra parameters
            """
            mp = parameters
            def _effective_diameter():
                """
                Calculates the effective rotor diameter
                """
                return ((mp["radius"]*2)*(
                        (1. + (1. - mp["thrust_coeff"])**0.5)/
                        (2.*(1. - mp["thrust_coeff"])**0.5))**0.5)

            def _r_nb():
                """
                Calculates R_nb
                """
                val1 = 1.08*mp["radius"]*2
                val2 = val1 + 21.7*mp["radius"]*2*(mp["ambient_intensity"]-0.05)
                return max([val1, val2])
            
            def _r_9_point_5(r_nb):
                """
                Calculates the wake radius at a distance of 9.5 rotor diameters
                downstream of the turbine
                """
                return 0.5*(r_nb + min([mp["hub_height"], r_nb]))

            def _x0(r_9_point_5, effective_diameter):
                """
                Calculates the position of the rotor w.r.t the applied
                coordinate system
                """
                return ((9.5*mp["radius"]*2)/
                        ((2.*r_9_point_5/effective_diameter)**3 - 1))
                
            def _rotor_disc_area():
                """
                Calculates the rotor disc area
                """
                return numpy.pi*(mp["radius"]**2)
            
            def _prandtl_mixing(effective_diameter, rotor_disc_area, x0):
                """
                Calculates the prandtl mixing length
                """
                return (((effective_diameter/2.)**(5./2.))*
                        ((105./(2*numpy.pi))**(-1./2.))*
                        ((mp["thrust_coeff"]*rotor_disc_area*x0)**(-5./6.)))

            effective_diameter = _effective_diameter()
            r_nb = _r_nb()
            r_9_point_5 = _r_9_point_5(r_nb)
            x0 = _x0(r_9_point_5, effective_diameter)
            rotor_disc_area = _rotor_disc_area()
            prandtl_mixing = _prandtl_mixing(effective_diameter,
                                             rotor_disc_area)

            params = {"effective_diameter": effective_diameter, 
                                    "r_nb": r_nb, 
                             "r_9_point_5": r_9_point_5, 
                                      "x0": x0, 
                         "rotor_disc_area": rotor_disc_area, 
                          "prandtl_mixing": prandtl_mixing}
            return params


        # set the default required parameters
        default_parameters = {"thrust_coeff": 0.6, "ambient_intensity": 0.1,
                              "hub_height": 10.}
        model_parameters = self._check_parameters(model_parameters,
                                                  default_parameters)
        model_parameters.update({"radius": turbine_radius})
        model_parameters.update(_setup(model_parameters))

        # initialize a Model object with these parameters
        super(Larsen, self).__init__("Larsen", flow_field, model_parameters)


    def wake_radius(self, turbine, point):
        """
        Returns the wake radius at point given that point is in the wake of
        turbine
        """
        dist = self.distance_between(turbine, point)
        mp = self.model_parameters
        return ((35./(2*numpy.pi))**(1/5.)*
                (3*(mp["prandtl_mixing"]**2))**(1/5.)*
                (mp["thrust_coeff"]*mp["rotor_disc_area"]*
                    (dist + mp["x0"]))**(1/3.))


    def individual_factor(self, turbine, point):
        """
        Returns the individual velocity reduction factor
        """
        mp = self.model_parameters
        x = self.distance_between(turbine, point) 
        r = self.dist_from_wake_center(turbine, point)
        xx0 = x + mp["x0"]
        bracket1 = mp["thrust_coeff"]*mp["rotor_disc_area"]*xx0**(-2.)
        bracket2 = (3*(mp["prandtl_mixing"]**2)*mp["thrust_coeff"]*
                    mp["rotor_disc_area"]*xx0)
        bracket3 = (17.5/numpy.pi)
        bracket4 = 3*(mp["prandtl_mixing"]**2)
        try:
            ret = 1. - (1./9)*(bracket1**(1./3))*(r**(1.5)*bracket2**(-0.5) -
                (bracket3**0.3)*(bracket4**-0.2))**2
        # caused when finding derivatives when r=0 so remove r
        except ZeroDivisionError: 
            ret = 1. - ((1./9)*(bracket1**(1./3))*
                        (-(bracket3**0.3)*(bracket4**-0.2))**2)
        return ret


    def get_search_radius(self, recovery_loss=2.5):
        """
        Returns the search radius for an acceptable recovery loss
        """
        # check if recovery_loss is zero
        if (recovery_loss < 1e-8):
            return numpy.inf
        else:
            recovery_loss /= 100.
            mp = self.model_params
            alpha = mp["thrust_coeff"]*mp["rotor_disc_area"]
            beta0 = -(17.5/numpy.pi)**(3./10)
            beta1 = (3*mp["prandtl_mixing"]**2)**(-1./5)
            beta = (beta0*beta1)**2
            return (alpha/((9*(1-recovery_loss))/beta)**3)**0.5
