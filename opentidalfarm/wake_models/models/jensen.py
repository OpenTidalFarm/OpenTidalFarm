from model import Model
import numpy

class Jensen(Model):
    """
    Defines the Jensen wake model
    """
    def __init__(self, flow_field, turbine_radius, model_parameters=None):
        # set the default required parameters
        default_parameters = {"thrust_coeff": 0.6, "wake_decay": 0.03}
        # check the given parameters against the default -- if one is missing
        # use the defined default value for it
        model_parameters = self._check_parameters(model_parameters,
                                                  default_parameters)
        model_parameters.update({"radius": turbine_radius})
        # initialize a Model object with these parameters
        super(Jensen, self).__init__("Jensen", flow_field, model_parameters)
    

    def wake_radius(self, turbine, point):
        """
        Returns the wake radius at point given that point is in the wake of
        turbine
        """
        dist = self.distance_between(turbine, point)
        mp = self.model_parameters
        return mp["radius"]*(1. + 2*mp["wake_decay"]*(dist/(2*mp["radius"])))


    def individual_factor(self, turbine, point):
        """
        Returns the individual velocity reduction factor
        """
        mp = self.model_parameters
        dist = self.distance_between(turbine, point)
        return ((1. - (1. - mp["thrust_coeff"])**0.5/ ((1. +
                mp["wake_decay"]*dist/mp["radius"])**2)))


    def get_search_radius(self, recovery_loss=2.5):
        """
        Returns the search radius for an acceptable recovery loss
        """
        # check if recovery_loss is zero
        if (recovery_loss < 1e-8):
            return numpy.inf
        else:
            recovery_loss /= 100.
            mp = self.model_parameters
            sq = (mp["thrust_coeff"]/recovery_loss)**0.5
            return (-1 + sq)/(mp["wake_decay"]/mp["radius"])
