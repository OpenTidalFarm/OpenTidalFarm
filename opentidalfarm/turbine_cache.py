import numpy
from dolfin import *
from dolfin_adjoint import *
from options import options
from turbine_function import TurbineFunction

class TurbineCache(object):

    def __init__(self, turbine_function_space=None):
        self.cache = {"turbine_pos": [],
                      "turbine_friction": []}
        self._function_space = None
        self._turbine_specification = None
        self._parameterisation = None
        self._controlled_by = None


    def _set_turbine_specification(self, turbine_specification):
        self._turbine_specification = turbine_specification
        self._parameterisation = turbine_specification.parameterisation
        self._controlled_by = turbine_specification.controls


    def _deserialize(self, serialized):
        """Deserializes the current parameter set.

        :param serialized: Serialized turbine paramaterisation of length 1, 2,
            or 3 times the number of turbines in the farm.
        :type serialized: numpy.ndarray.
        :raises: ValueError.

        """
        frictions = []
        positions = []

        # Turbine_friction and turbine_pos are control parameters
        if self._controlled_by.friction and self._controlled_by.position:
            # Break up the input parameters into friction and tuples of
            # coordinates.
            frictions = serialized[:len(serialized)/3]
            positions = serialized[len(serialized)/3:]

        # Only turbine_friction is a control parameter.
        elif self._controlled_by.friction:
            frictions = serialized
        # Only the turbine_pos is a control parameter.
        elif self._controlled_by.position:
            positions = serialized

        return positions, frictions


    def _unflatten_positions(self, positions):
        return [(positions[2*i], positions[2*i+1])
                for i in xrange(len(positions)/2)]


    def update(self, m):
        """Creates a list of all turbine function/derivative interpolations.
        This list is used as a cache to avoid the recomputation of the expensive
        interpolation of the turbine expression."""

        if self._turbine_specification is None:
            raise ValueError("The turbine specification must be set before "
                             "caching is possible.")

        # Deserialize the parameters.
        flattened_positions, frictions = self._deserialize(m)
        positions = self._unflatten_positions(flattened_positions)

        # If the parameters have not changed, then there is no need to do
        # anything.
        if "turbine_pos" in self.cache and "turbine_friction" in self.cache:
            if (len(positions) == len(self.cache["turbine_pos"]) and
                len(frictions) == len(self.cache["turbine_friction"]) and
                (frictions == self.cache["turbine_friction"]).all() and
                (positions == self.cache["turbine_pos"]).all()):
                return

        if self._parameterisation.smeared:
            tf = Function(self._function_space, name="turbine_friction_cache")
            optimization.set_local(tf, frictions)
            self.cache["turbine_field"] = tf
            return

        log(INFO, "Updating turbine cache")

        # Store the new turbine parameters.
        if len(positions) > 0:
            self.cache["turbine_pos"] = numpy.copy(positions)
        if len(frictions) > 0:
            self.cache["turbine_friction"] = numpy.copy(frictions)


        # Precompute the interpolation of the friction function of all turbines
        turbines = TurbineFunction(self._function_space,
                                   self._turbine_specification)

        if self._controlled_by.dynamic_friction:
            # If the turbine friction is controlled dynamically, we need to
            # cache the turbine field for every timestep
            self.cache["turbine_field"] = []
            for t in xrange(len(frictions)):
                tf = turbines(self.cache,
                              name="turbine_friction_cache_t_" + str(t),
                              timestep=t)
                self.cache["turbine_field"].append(tf)
        else:
            tf = turbines(self.cache, name="turbine_friction_cache")
            self.cache["turbine_field"] = tf


        # Precompute the interpolation of the friction function for each
        # individual turbine.
        if options["output_individual_power"]:
            log(INFO, ("Building individual turbine power friction functions "
                       "for caching purposes..."))
            self.cache["turbine_field_individual"] = []
            for i in xrange(len(frictions)):
                cpy_cache = {"turbine_pos": [positions[i]],
                             "turbine_friction": [frictions[i]]}
                tf = turbines(cpy_cache)
                self.cache["turbine_field_individual"].append(tf)

        # Precompute the derivatives with respect to the friction magnitude of
        # each turbine.
        if self._controlled_by.friction:
            self.cache["turbine_derivative_friction"] = []
            for n in xrange(len(frictions)):
                tfd = turbines(self.cache,
                               derivative_index=n,
                               derivative_var='turbine_friction',
                               name=("turbine_friction_derivative_with_respect_"
                                     "friction_magnitude_of_turbine_" +
                                     str(n)))
                self.cache["turbine_derivative_friction"].append(tfd)

        elif self._controlled_by.dynamic_friction:
            self.cache["turbine_derivative_friction"] = []
            for t in xrange(len(frictions)):
                self.cache["turbine_derivative_friction"].append([])
                for n in range(len(frictions[t])):
                    tfd = turbines(self.cache,
                                   derivative_index=n,
                                   derivative_var="turbine_friction",
                                   name=("turbine_friction_derivative_with_"
                                         "respect_friction_magnitude_of_"
                                         "turbine_" + str(n) + "t_" + str(t)),
                                   timestep=t)
                    self.cache["turbine_derivative_friction"][t].append(tfd)

        # Precompute the derivatives with respect to the turbine position.
        if self._controlled_by.position:
            if not self._controlled_by.dynamic_friction:
                self.cache["turbine_derivative_pos"] = []
                for n in xrange(len(positions)):
                    self.cache["turbine_derivative_pos"].append({})
                    for var in ('turbine_pos_x', 'turbine_pos_y'):
                        tfd = turbines(self.cache,
                                       derivative_index=n,
                                       derivative_var=var,
                                       name=("turbine_friction_derivative_"
                                             "with_respect_position_of_turbine_"
                                             + str(n)))
                        self.cache["turbine_derivative_pos"][-1][var] = tfd
            else:
                self.cache["turbine_derivative_pos"] = []
                for t in xrange(len(friction)):
                    self.cache["turbine_derivative_pos"].append([])
                    for n in xrange(len(positions)):
                        self.cache["turbine_derivative_pos"][t].append({})
                        for var in ('turbine_pos_x', 'turbine_pos_y'):
                            tfd = turbines(self.cache,
                                           derivative_index=n,
                                           derivative_var=var,
                                           name=("turbine_friction_derivative_"
                                                 "with_respect_position_of_"
                                                 "turbine_" + str(n)),
                                           timestep=t)
                            self.cache["turbine_derivative_pos"][t][-1][var] \
                                    = tfd
