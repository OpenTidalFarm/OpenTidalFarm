import numpy
from dolfin import *
from dolfin_adjoint import *
from turbine_function import TurbineFunction
from options import options

class TurbineCache(dict):
    def __init__(self, *args, **kw):
        super(TurbineCache, self).__init__(*args, **kw)
        self.itemlist = super(TurbineCache, self).keys()
        self._function_space = None
        self._specification = None
        self._controlled_by = None
        self["position"] = []
        self["friction"] = []

    def __setitem__(self, key, value):
        self.itemlist.append(key)
        super(TurbineCache,self).__setitem__(key, value)

    def __iter__(self):
        return iter(self.itemlist)

    def keys(self):
        return self.itemlist

    def values(self):
        return [self[key] for key in self]

    def itervalues(self):
        return (self[key] for key in self)

    def set_function_space(self, function_space):
        self._function_space = function_space

    def set_turbine_specification(self, specification):
        self._specification = specification
        self._controlled_by = specification.controls

    def _add_turbines(self, position=None, friction=None):
        # Ensure we don't use this with the smeared approach.
        assert(not self._specification.smeared)
        # And that we have the correct size inputs.
        assert(len(position)==len(friction))
        self["position"] += position
        self["friction"] += friction


    def update(self, farm):
        """Creates a list of all turbine function/derivative interpolations.
        This list is used as a cache to avoid the recomputation of the expensive
        interpolation of the turbine expression."""

        try:
            assert(self._specification is not None)
        except AssertionError:
            raise ValueError("The turbine specification has not yet been set.")

        position = farm._parameters["position"]
        friction = farm._parameters["friction"]

        # If the parameters have not changed, then there is no need to do
        # anything.
        if (len(friction)==len(self["friction"]) and
            len(position)==len(self["position"]) and
            (numpy.asarray(friction)==numpy.asarray(self["friction"])).all() and
            (numpy.asarray(position)==numpy.asarray(self["position"])).all()):
            return

        # Update the cache.
        log(INFO, "Updating the turbine cache")

        # Update the positions and frictions.
        self["friction"] = numpy.copy(friction)
        self["position"] = numpy.copy(position)

        # For the smeared approached we just update the turbine_field.
        if self._specification.smeared:
            tf = Function(self._function_space, name="turbine_friction_cache")
            optimization.set_local(tf, self["friction"])
            self["turbine_field"] = tf
            return


        # Precompute the interpolation of the friction function of all turbines.
        turbines = TurbineFunction(self, self._function_space,
                                   self._specification)

        # If the turbine friction is controlled dynamically, we need to cache
        # the turbine field for every timestep.
        if self._controlled_by.dynamic_friction:
            self["turbine_field"] = []
            for t in xrange(len(self["friction"])):
                tf = turbines(name="turbine_friction_cache_t_"+str(t),
                              timestep=t)
                self["turbine_field"].append(tf)
        else:
            tf = turbines(name="turbine_friction_cache")
            self["turbine_field"] = tf

        # Precompute the interpolation of the friction function for each
        # individual turbine.
        if options["output_individual_power"]:
            log(INFO, "Building individual turbine power friction functions "
                      "for caching purposes...")
            self["turbine_field_individual"] = []
            for i in xrange(len(self["friction"])):
                position_cpy = [self["position"][i]]
                friction_cpy = [self["friction"][i]]
                turbine = TurbineFunction(self, self, self._function_space,
                                          self._specification)
                tf = turbine()
                self["turbine_field_individual"].append(tf)

        # Precompute the derivatives with respect to the friction magnitude
        # of each turbine.
        if self._controlled_by.friction:
            self["turbine_derivative_friction"] = []
            for n in xrange(len(self["friction"])):
                tfd = turbines(derivative_index=n,
                               derivative_var="turbine_friction",
                               name=("turbine_friction_derivative_with_"
                                     "respect_friction_magnitude_of_turbine_"
                                     + str(n)))
                self["turbine_derivative_friction"].append(tfd)

        elif self._controlled_by.dynamic_friction:
            self["turbine_derivative_friction"] = []
            for t in xrange(len(self["friction"])):
                self["turbine_derivative_friction"].append([])

                for n in xrange(len(self["friction"][t])):
                    tfd = turbines(derivative_index=n,
                                   derivative_var="turbine_friction",
                                   name=("turbine_friction_derivative_with_"
                                         "respect_friction_magnitude_of_"
                                         "turbine_" + str(n) + "t_" +
                                         str(t)),
                                   timestep=t)
                    self["turbine_derivative_friction"][t].append(tfd)

        # Precompute the derivatives with respect to the turbine position.
        if self._controlled_by.position:
            if not self._controlled_by.dynamic_friction:
                self["turbine_derivative_pos"] = []
                for n in xrange(len(self["position"])):
                    self["turbine_derivative_pos"].append({})
                    for var in ("turbine_pos_x", "turbine_pos_y"):
                        tfd = turbines(derivative_index=n, derivative_var=var,
                                       name=("turbine_friction_derivative_"
                                             "with_respect_position_of_"
                                             "turbine_" + str(n)))
                        self["turbine_derivative_pos"][-1][var] = tfd
            else:
                self["turbine_derivative_pos"] = []
                for t in xrange(len(self["friction"])):
                    self["turbine_derivative_pos"].append([])

                    for n in xrange(len(self["position"])):
                        self["turbine_derivative_pos"][t].append({})
                        for var in ("turbine_pos_x", "turbine_pos_y"):
                            tfd = turbines(derivative_index=n,
                                           derivative_var=var,
                                           name=("turbine_friction_"
                                                 "derivative_with_respect_"
                                                 "position_of_turbine_" +
                                                 str(n)),
                                           timestep=t)
                            self["turbine_derivative_pos"][t][-1][var] = tfd
