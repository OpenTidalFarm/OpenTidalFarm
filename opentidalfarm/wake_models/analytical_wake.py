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
    def __init__(self, config, flow_field, model_type='ApproximateShallowWater',
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
            self.model = valid_models[model_type](flow_field,
                                                  turbine_radius,
                                                  model_params)

        # flat to tell functions when to use ad.adnumber objects -- switched on
        # and off in grad and hess
        self.diff = False


    def eval(self, values, x):
        """
        Evaluates the flow field at x and stores the result in values
        """
        raise NotImplementedError("Since the restructure to take only take "
            "turbines within a certain radius to calculate the power, this has "
            "not been updated")
        #values[:] = factor*numpy.array([self.model.flow_x(x), self.model.flow_y(x)])


    def value_shape(self):
        """
        Returns shape of value
        """
        return (2,)


    def use_ad(function):
        """
        Decorator function to make a function switch on the self.diff flag (to
        use ad.adnumber objects) before calling the wrapped function then
        switching off the flag after
        """
        def wrapped(self, *args, **kwargs):
            self.diff=True
            ret = function(self, *args, **kwargs)
            self.diff=False
            return ret
        return wrapped


    @use_ad
    def grad(self, turbines, acceptable_loss=2.5):
        """
        Returns the gradient of the power wrt to turbine coordinates. Turbines
        should be a flattened numpy array.
        """
        turbines = ad.adnumber(turbines)
        tot_pow = self._total_power(turbines, acceptable_loss=acceptable_loss)
        return numpy.array(tot_pow.gradient(turbines))


    @use_ad
    def hess(self, turbines, acceptable_loss=2.5):
        """
        Returns the hessian of the power wrt to turbine coordinates
        """
        turbines = ad.adnumber(turbines)
        tot_pow = self._total_power(turbines, acceptable_loss=acceptable_loss)
        return numpy.array(tot_pow.hessian(turbines))


    def _individual_power(self, turbines, index, indices_to_check):
        """
        Returns the power of turbines[index] due to the wake of the turbines at
        the indices_to_check
        """
        turbines_to_check = [turbines[i] for i in indices_to_check]
        flow_velocity = self._flow_magnitude_at(turbines[index])
        reduction_factor = self._combined_factor(turbines[index],
                                                 turbines_to_check)

        return self._power_of(flow_velocity*reduction_factor)


    def _total_power(self, turbines, acceptable_loss=2.5):
        """
        Returns the total power output of the turbine array; turbines should be
        a flattened array
        """
        def _tupleize(turbines):
            """
            Convenience function; turns a flattened array into tuples
            """
            return [(turbines[i*2], turbines[i*2+1]) for i in range(len(turbines)/2)]

        if self.diff:
            # create non-ad list of turbines
            non_ad = numpy.array([t.x for t in turbines])
            non_ad = _tupleize(non_ad)
            turbines = _tupleize(turbines)
        else:
            turbines = _tupleize(turbines)
            non_ad = turbines

        # get a search radius based on an acceptable recovery loss percentage
        radius = self.model.get_search_radius(acceptable_loss)
        # get the indicies of turbines within the search radius
        in_radius_indices = self._compute_within_radius(non_ad, radius)
        # generate a list of turbines to check each turbine against based on
        # whether or not they are in wake
        to_check = self._compute_within_wake(non_ad, in_radius_indices)
        total = 0.0
        for i in range(len(to_check)):
            # if the turbines[i] is in the wake of other turbines, calculate the
            # power at that point due to the combined wakes
            if len(to_check[i]) > 0:
                ind = self._individual_power(turbines, i, to_check[i])

            # else there is no reduction factor so we can take the power of the
            # flow magnitude at that point
            else:
                ind = self._power_of(self._flow_magnitude_at(turbines[i])
                                       *self.model.individual_factor(0,0))
            total += ind
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
            y0 = abs(self.model.dist_from_wake_center(turbine, point))
            x0 = abs(self.model.distance_between(turbine, point))
            return y0 < self.model.wake_radius(x0)
        # Approximate shallow water model causes wake upstream too
        elif self.model.model_type == "ApproximateShallowWater":
            x0 = abs(self.model.distance_between(turbine, point))
            y0 = abs(self.model.dist_from_wake_center(turbine, point))
            return ((x0 < self.model._upstream_wake()) and
                    (y0 < self.model.wake_radius(x0)))
        # if not downstream, then not in wake
        else:
            return False


    def _combined_factor(self, point, turbines):
        """
        Returns the combined factor of a model at point due to the wake from
        turbines
        """
        def _combine(factors):
            """
            Combines factors using a modified sum of squares which allows for
            factors greater than 1
            """
            ret = 1.
            for f in factors:
                ret = ret*f
            # stop turbines piling up to create an artifically high factor
            return min([ret, 1.25])

        factors = []
        for t in turbines:
            x0 = self.model.distance_between(t, point)
            y0 = self.model.dist_from_wake_center(t, point)
            factors.append(self.model.individual_factor(x0,y0))

        # include the effect a turbine has on itself
        return _combine(factors)*self.model.individual_factor(0,0)


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
        velocity_for_rated_power = 3.5
        reduced_velocity = velocity_for_rated_power*self.model.individual_factor(0,0)
        rated_power = 3.5e6
        factor = rated_power/reduced_velocity**3
        power = factor*speed**3
        return power 


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


    def _compute_within_wake(self, turbine_tuples, ind):
        """
        Returns a list whose values represent which turbines to check when
        calculating an individual reduction factor.

        I.e. for a list of turbines we return the turbines to check at each
             corresponding index:
                turbines =  t0,   t1,      t2, ..., tn
                to_check =  [], [t0], [t1,t3], ..., []

        ind is a list of index tuples of turbines that are within a given radius
        of each other
        """
        to_check = [[] for i in range(len(turbine_tuples))]
        # when checking the individual factor of a point we want to be able to
        # iterate over a list which contains lists of turbines to check -- None
        # indicates that this turbine is not in the wake of any others
        for i in range(len(ind)):
            t0 = ind[i][0]
            t1 = ind[i][1]
            # t1 in wake of t0
            if self._is_in_wake(turbine_tuples[t0], turbine_tuples[t1]):
                to_check[ind[i][1]].append(t0)
                # also possible for t0 to be in wake of t1
                if self._is_in_wake(turbine_tuples[t1], turbine_tuples[t0]):
                    to_check[ind[i][0]].append(t1)
            # t0 in wake of t1
            elif self._is_in_wake(turbine_tuples[t1], turbine_tuples[t0]):
                to_check[ind[i][0]].append(t1)
            else:
                pass
        return to_check
