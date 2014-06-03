from ..math_helpers import l2_norm, vector_difference, normalize_vector
from .. import helpers
import numpy

class Model(object):
    """
    A superclass which defines wake models.
    
    To define a wake model (e.g. like Jensen) it must have the following
    methods:

        __init__(self, flow_field, turbine_radius, model_parameters=None)
            If model_parameters should be checked against default parameters
            (which should be defined in a dictionary in this method) and check
            with the _check_parameters method. 

            turbine_radius should also be added to model_parameters with the key
            "radius". The base class should then be instantiated with:
            
            super(Jensen, self).__init__(MODEL_NAME, flow_field, model_parameters)

        wake_radius(self, turbine, point)
            Should return the radius of the wake at point in the wake of turbine

        individual_factor(self, turbine, point)
            If point is in the wake of turbine this should return the velocity
            reduction factor between 0 and 1

        get_search_radius(self, recovery_loss=DEFAULT_VALUE)
            Returns the search distance where:
                individual_factor == (1 - recovery_loss/100.)
            A default value should be set; usually 2.5(%) but if the model
            defines a distance where recovery is complete then it should be set
            to 0.0 and the recovery distance should be returned.
    """
    def __init__(self, model_type, flow_field, model_parameters):
        self.model_type = model_type
        self.__model_parameters__ = model_parameters
        # for AD we need to split the flow field into two seperate x and y
        # fields -- if only one field then we get problems with performing
        # operations on arrays of ad.ADF objects
        if "compute_gradient" in self.__model_parameters__:
            if "flow_gradient" in self.__model_parameters__:
                flow_gradient = self.__model_parameters__["flow_gradient"]
            else:
                flow_gradient = None
            self.flow_x = helpers.ADDolfinVecX(flow_field,
                                  self.__model_parameters__["compute_gradient"],
                                  flow_gradient)
            self.flow_y = helpers.ADDolfinVecY(flow_field,
                                  self.__model_parameters__["compute_gradient"],
                                  flow_gradient)
        else:
            self.flow_x = helpers.ADDolfinVecX(flow_field)
            self.flow_y = helpers.ADDolfinVecY(flow_field)
        

    def __repr__(self):
        title = "%s model with the following parameters" % (self.model_type)
        for key in self.__model_parameters__:
            title += "\n%s = %s" % (key, self.__model_parameters__[key])
        return title


    def _check_parameters(self, model_parameters, default_parameters):
        """
        Checks given model parameters against default parameters, updates model
        parameters if a value is given
        """
        if model_parameters is None:
            model_parameters = default_parameters
        else:
            for key in default_parameters:
                if key not in model_parameters:
                    model_parameters.update({key: default_parameters[key]})
        return model_parameters
    


    def distance_between(self, turbine, point):
        """
        Returns distance between turbine and point parallel to the direction of
        flow at turbine
        """
        diff = vector_difference(point, turbine)
        flow = numpy.array([self.flow_x(turbine), self.flow_y(turbine)])
        v1 = normalize_vector(diff)
        v2 = normalize_vector(flow)
        cos_theta = (v1[0]*v2[0] + v1[1]*v2[1])
        return l2_norm(diff)*cos_theta


    def dist_from_wake_center(self, turbine, point):
        """
        Returns the distance of point from the centerline of the wake generated
        by turbine
        """
        flow = numpy.array([self.flow_x(turbine), self.flow_y(turbine)])
        slope = flow[1]/(flow[0]+1e-16) # dy/dx; avoid zero division
        cross = turbine[1] - slope*turbine[0] # c = y - slope*x
        ret = (point[1] - slope*point[0] - cross)/((slope**2 + 1)**0.5)
        return ret
