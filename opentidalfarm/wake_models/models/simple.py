from model import Model
from ..math_helpers import k_fac, box_car, heaviside

class Simple(Model):
    """
    Defines the Jensen wake model
    """
    def __init__(self, flow_field, turbine_radius, model_parameters=None):
        # set the default required parameters
        default_parameters = {"recovery_distance": 60*turbine_radius,
                              "f0": 1.3,
                              "f1": 0.2, 
                              "f2": 0.45}
        default_parameters.update({"max_wake_radius": 
                                   0.3*default_parameters["recovery_distance"]})
        # check the given parameters against the default -- if one is missing
        # use the defined default value for it
        model_parameters = self._check_parameters(model_parameters,
                                                  default_parameters)
        model_parameters.update({"radius": turbine_radius})
        # initialize a Model object with these parameters
        super(Simple, self).__init__("Simple", flow_field, model_parameters)
    

    def wake_radius(self, turbine, point):
        """
        Returns the wake radius at point given that point is in the wake of
        turbine
        """
        dist = self.distance_between(turbine, point)
        mp = self.model_parameters
        return dist**(1./3)*((mp["max_wake_radius"] - mp["radius"])/
               (mp["recovery_distance"]**(1./3))) + mp["radius"]


    def individual_factor(self, turbine, point):
        """
        Returns the individual velocity reduction factor
        """
        mp = self.model_parameters
        x = self.distance_between(turbine, point) 
        y = self.dist_from_wake_center(turbine, point)
        r = self.wake_radius(turbine, point)
        k = k_fac(r, mp["radius"], mp["max_wake_radius"]) 
        m = 1./mp["f0"]
        a = mp["f1"]*r
        b = mp["f2"]*r
        reduction = box_car(y, -b, b, k)*(1-m) + m
        unscaled = (mp["f0"] - box_car(y, -a, a, k))*reduction
        return unscaled + (1 - unscaled)*heaviside(x, 
                                                   mp["recovery_distance"]/2., 
                                                   5.75/mp["recovery_distance"])
    
    def get_search_radius(self, recovery_loss=None):
        """
        Returns the search radius for an acceptable recovery loss
        """
        return self.model_parameters["recovery_distance"]
