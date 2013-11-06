from dolfin import Expression
from math_helpers import vector_difference, l2_norm, angle_between_vectors
import numpy
import models
import inspect
import sys
import ad
import ad.admath
import itertools

class AnalyticalWake(Expression):
    # TODO: sort out this docstring
    """
    An Analytical Wake class.
    """
    def __init__(self, config, flow_field, model_type='Jensen',
                 model_params=None):

        # make a dict of the available model types: {"model_name": model_obj}
        # gets data from looking for classses in the models submodule
        valid_models = dict(inspect.getmembers(sys.modules[models.__name__],
                                               inspect.isclass))
        # check model type is valid
        if model_type not in valid_models:
            raise NotImplementedError("Model type is invalid or has not been "
                                      "impplemented yet")
        else:
            turbine_radius = (config.params["turbine_x"] +
                              config.params["turbine_y"])/4.
            self.model = valid_models[model_type](flow_field, turbine_radius, model_params)
        
        if isinstance(config.params["turbine_pos"], numpy.ndarray):
            self.turbines = config.params["turbine_pos"].flatten()
        else:
            self.turbines = numpy.array(config.params["turbine_pos"]).flatten()

        self.config = config


    def eval(self, values, x):
        """
        Evaluates the flow field at x and stores the result in values
        """
        raise NotImplementedError("Since the restructure to take only take "
            "turbines within a certain radius to calculate the power, this has "
            "not been updated")
        #factor = self._combined_factor(x, self.turbines)
        #values[:] = factor*numpy.array([self.model.flow_x(x), self.model.flow_y(x)])


    def value_shape(self):
        """
        Returns shape of value
        """
        return (2,)


    def grad(self, turbines=None, acceptable_loss=2.5):
        """
        Returns the gradient of the power wrt to turbine coordinates
        """
        # update the turbines if provided   
        if turbines is not None: 
            self.update_turbines(turbines)
        turbines = ad.adnumber(turbines)
        tot_pow = self._total_power(turbines, acceptable_loss=acceptable_loss)
        return numpy.array(tot_pow.gradient(turbines))


    def hess(self, turbines=None, acceptable_loss=2.5):
        """
        Returns the hessian of the power wrt to turbine coordinates
        """
        # update the turbines if provided   
        if turbines is not None: 
            self.update_turbines(turbines)
        turbines = ad.adnumber(turbines)
        tot_pow = self._total_power(turbines, acceptable_loss=acceptable_loss)
        return numpy.array(tot_pow.hessian(turbines))


    def update_turbines(self, turbines):
        """
        Updates the turbines stored in self.turbines
        """
        self.turbines = numpy.array(turbines).flatten()


    def _individual_power(self, point, turbines):
        """
        Returns the individual power of a turbine at point; turbines should be a
        flattened array
        """
        flow_velocity = self._flow_magnitude_at(point)
        reduction_factor = self._combined_factor(point, turbines)
        return self._power_of(flow_velocity*reduction_factor)

    
    def _total_power(self, turbines, acceptable_loss=2.5):
        """
        Returns the total power output of the turbine array; turbines should be
        a flattened array
        """
        # update turbines with the given turbines and then turn the list into
        # tuples for convenience
        self.update_turbines(turbines)
        turbines = numpy.array([(self.turbines[i*2], self.turbines[i*2+1])
                                for i in range(len(self.turbines)/2)])

        # get a search radius based on an acceptable recovery loss percentage
        radius = self.model.get_search_radius(acceptable_loss)
        # get the indicies of turbines within the search radius
        indices = self._compute_within_radius(turbines, radius) 
        # generate a list of turbines to check each turbine against based on
        # whether or not they are in wake
        to_check = self._compute_within_wake(indices)
        total = 0.0
        for i in range(len(to_check)):
            # if the turbines[i] is in the wake of other turbines, calculate the
            # power at that point due to the combined wakes
            if to_check[i] is not None:
                total += self._individual_power(turbines[i], to_check[i])
            # else there is no reduction factor so we can take the power of the
            # flow magnitude at that point
            else:
                total +=  self._power_of(self._flow_magnitude_at(turbines[i]))
        return total


    def _is_in_wake(self, turbine, point):
        """
        Returns whether or not point is in the wake of turbine
        """
        def _is_downstream(self, turbine, point):
            """
            Returns whether or not point is downstream of turbine
            """
            diff = vector_difference(point, turbine)
            flow = (self.model.flow_x(turbine), self.model.flow_y(turbine))
            angle = angle_between_vectors(diff, flow)
            return (angle < ad.admath.pi/2)

        # if downstream, compare distance from centerline of wake to wake radius
        if _is_downstream(self, turbine, point):
            dist_from_wake_center = self.model.dist_from_wake_center(turbine, point)
            return dist_from_wake_center < self.model.wake_radius(turbine, point)
        # if not downstream, then not in wake
        else:
            return False

    
    def _combined_factor(self, point, turbines):
        """
        Returns the combined factor of a model at point due to the wake from
        turbines
        """
        factor = 0.0
        for t in turbines:
            factor += (1. - self.model.individual_factor(t, point))**2
        return 1 - factor**0.5
    

    def _flow_magnitude_at(self, x):
        """
        Returns the magnitude of the flow at x
        """
        flow = numpy.array([self.model.flow_x(x), self.model.flow_y(x)])
        return l2_norm(flow)


    def _power_of(self, speed):
        """
        Returns the power given the speed
        """
        fac = 1.5e6/(3**3)
        power = fac*speed**3
        return power if power < 1.5e6 else 1.5e6

   
    def _within_radius(self, turbine, point, radius):
        """
        Returns true if point lies within the radius of turbine
        """
        diff = vector_difference(point, turbine)
        return l2_norm(diff) < radius

    
    def _compute_within_radius(self, turbine_tuples, radius=300.):
        """
        Returns a list of indices for turbine pairs within a given radius of
        each other
        """
        ind = list(itertools.combinations(range(len(turbine_tuples)), 2))
        if radius == numpy.inf:
            return ind
        else:
            def _filter(index):
                """
                Convenience method so we can use itertools.ifilter with a single
                argument
                """
                return self._within_radius(turbine_tuples[index[0]], 
                                           turbine_tuples[index[1]], 
                                           radius)
            return list(itertools.ifilter(_filter, ind))


    def _compute_within_wake(self, ind):
        """
        Returns a list whose values represent which turbines to check when
        calculating an individual reduction factor.

        I.e. for a list of turbines we return the turbines to check at each
             corresponding index:
                turbines =   t0,   t1,      t2, ...,   tn
                to_check = None, [t0], [t1,t3], ..., None

        ind is a list of index tuples of turbines that are within a given radius
        of each other
        """
        # when checking the individual factor of a point we want to be able to
        # iterate over a list which contains lists of turbines to check -- None
        # indicates that this turbine is not in the wake of any others
        to_check = [None]*(len(self.turbines)/2)
        for i in range(len(ind)):
            # now working with a flattened list of turbine positions so need to
            # create tuples
            t0 = (self.turbines[ind[i][0]*2], self.turbines[ind[i][0]*2+1])
            t1 = (self.turbines[ind[i][1]*2], self.turbines[ind[i][1]*2+1])
            if self._is_in_wake(t0, t1):
                if to_check[ind[i][1]] is None:
                    to_check[ind[i][1]] = [t0]
                else:
                    to_check[ind[i][1]].append(t0)
            elif self._is_in_wake(t1, t0):
                if to_check[ind[i][0]] is None:
                    to_check[ind[i][0]] = [t1]
                else:
                    to_check[ind[i][0]].append(t1)
            else:
                pass
        return to_check
