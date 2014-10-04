import numpy
from parameter_dict import ParameterDictionary
from dolfin import *
from dolfin_adjoint import *
from output import output_options


class TurbineFunction(object):

    def __init__(self, V, turbine_prototype, derivative_index=-1):
        # self.params = ParameterDictionary(params)
        self._turbine_prototype = turbine_prototype

        # Precompute some turbine parameters for efficiency.
        self.x = interpolate(Expression("x[0]"), V).vector().array()
        self.y = interpolate(Expression("x[1]"), V).vector().array()
        self.V = V


    def __call__(self, cache, name="", derivative_index=None,
                 derivative_var=None, timestep=None):
        """If the derivative selector is i >= 0, the Expression will compute the
        derivative of the turbine with index i with respect to either the x or y
        coorinate or its friction parameter. """

        if derivative_index is None:
            turbine_pos = cache["turbine_pos"]
            if timestep is None:
                turbine_friction = cache["turbine_friction"]
            else:
                turbine_friction = cache["turbine_friction"][timestep]
        else:
            turbine_pos = [cache["turbine_pos"][derivative_index]]
            if timestep is None:
                turbine_friction = [cache["turbine_friction"][derivative_index]]
            else:
                turbine_friction = (
                    [cache["turbine_friction"][timestep][derivative_index]])

        # Infeasible optimisation algorithms (such as SLSQP) may try to evaluate
        # the functional with negative turbine_frictions. Since the forward
        # model would crash in such cases, we project the turbine friction
        # values to positive reals.
        turbine_friction = [max(0, f) for f in turbine_friction]

        ff = numpy.zeros(len(self.x))
        # Ignore division by zero.
        numpy.seterr(divide="ignore")
        eps = 1e-12
        for (x_pos, y_pos), friction in zip(turbine_pos, turbine_friction):
            radius = self._turbine_prototype.radius
            x_unit = numpy.minimum(
                numpy.maximum((self.x-x_pos)/(radius, eps-1), 1-eps))
            y_unit = numpy.minimum(
                numpy.maximum((self.y-y_pos)/(radius, eps-1), 1-eps))

            # Apply chain rule to get the derivative with respect to the turbine
            # friction.
            e = numpy.exp(-1/(1-x_unit**2)-1./(1-y_unit**2)+2)
            if derivative_index is None:
                ff += e*friction

            elif derivative_var == "turbine_friction":
                ff += e

            if derivative_var == "turbine_pos_x":
                ff += e*(-2*x_unit/((1.0-x_unit**2)**2))*friction*(-1.0/radius)

            elif derivative_var == "turbine_pos_y":
                ff += e*(-2*y_unit/((1.0-y_unit**2)**2))*friction*(-1.0/radius)

        # Reset numpy to warn for zero division errors.
        numpy.seterr(divide="warn")

        f = Function(self.V, name=name, annotate=False)
        f.vector().set_local(ff)
        f.vector().apply("insert")
        return f


class TurbineCache(object):

    def __init__(self, turbine_function_space=None):
        self.cache = {"turbine_pos": [],
                      "turbine_friction": []}
        self._function_space = None
        self._turbine_prototype = None
        self._parameterisation = None
        self._controlled_by = None


    def _set_turbine_prototype(self, prototype):
        self._turbine_prototype = prototype
        self._parameterisation = prototype.parameterisation
        self._controlled_by = prototype.controls


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


    def __call__(self, m):
        """Creates a list of all turbine function/derivative interpolations.
        This list is used as a cache to avoid the recomputation of the expensive
        interpolation of the turbine expression."""

        if self._turbine_prototype is None:
            raise ValueError("The turbine prototype must be set before caching"
                             "is possible.")

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
        self.cache["turbine_pos"] = numpy.copy(positions)
        self.cache["turbine_friction"] = numpy.copy(frictions)


        # Precompute the interpolation of the friction function of all turbines
        turbines = TurbineFunction(self._function_space,
                                   self._turbine_prototype)

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
        if output_options["individual_power"]:
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
        if self._controlled_by.turbine_friction:
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
        if self._controlled_by.positions:
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
