from model import Model
from ..math_helpers import box_car, heaviside

class Simple(Model):
    """
    Defines the Jensen wake model
    """
    def __init__(self, flow_field, turbine_radius, model_parameters=None):
        # set the default required parameters
        default_parameters = {"recovery_distance": 30*turbine_radius,
                              "f0": 1.2,
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
    

    def wake_radius(self, x0):
        """
        Returns the wake radius at point given that point is in the wake of
        turbine
        """
        mp = self.__model_parameters__
        return x0**(1./3)*((mp["max_wake_radius"] - mp["radius"])/
                (mp["recovery_distance"]**(1./3))) + mp["radius"]


    def individual_factor(self, x0, y0):
        """
        Returns the individual velocity reduction factor
        """
        def _k_fac(y0):
            """
            Returns a scaled value of k to be used with the heaviside
            function
            """
            return 0.1 - 0.05*((y0 - mp["radius"])/
                               (mp["max_wake_radius"] - mp["radius"]))

        mp = self.__model_parameters__
        r = self.wake_radius(x0)
        k = _k_fac(y0)
        m = 1./mp["f0"]
        a = mp["f1"]*r
        b = mp["f2"]*r
        y_component_1 = box_car(y0, -b, b, k)*(1-m) + m
        y_component_2 = (mp["f0"] - box_car(y0, -a, a, k))
        y_component = y_component_1*y_component_2
        x_component = heaviside(x0, 
                                mp["recovery_distance"]/2.,
                                5.75/mp["recovery_distance"])
        return y_component + (1 - y_component)*x_component
    
    def get_search_radius(self, recovery_loss=None):
        """
        Returns the search radius for an acceptable recovery loss
        """
        return self.__model_parameters__["recovery_distance"]
