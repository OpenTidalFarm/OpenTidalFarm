import numpy
import dolfin
from .minimum_distance_constraints import MinimumDistanceConstraints
from ..turbine import Turbine
from ..turbine_cache import TurbineCache

class BaseFarm(object):
    """A base Farm class from which other Farm classes should be derived."""
    def __init__(self):
        """Create an empty Farm."""

        self._turbine_prototype = None
        # Create a chaching object for the interpolated turbine friction fields
        # (as their computation is very expensive)
        self.turbine_cache = TurbineCache()


    def _get_turbine_prototype(self):
        if self._turbine_prototype is None:
            raise ValueError("The prototype turbine has not yet been set.")
        return self._turbine_prototype


    def _set_turbine_prototype(self, prototype):
        self._turbine_prototype = prototype
        self.turbine_cache._set_turbine_prototype(self._turbine_prototype)


    turbine_prototype = property(_get_turbine_prototype,
                                 _set_turbine_prototype,
                                 "The prototype turbine specification.")

    @property
    def number_of_turbines(self):
        """The number of turbines in the farm.
        :returns: The number of turbines in the farm.
        :rtype: int
        """
        return len(self.turbine_cache.cache["turbine_pos"])


    @property
    def as_parameter_array(self):
        """A serialized representation of the farm based on the controls.

        :returns: A serialized representation of the farm based on the controls.
        :rtype: numpy.ndarray
        """
        m = []

        if self._turbine_prototype.controls.friction:
            for friction in self.turbine_cache.cache["turbine_friction"]:
                m.append(friction)
        if self._turbine_prototype.controls.position:
            for position in self.turbine_cache.cache["turbine_pos"]:
                m.append(position[0])
                m.append(position[1])

        return numpy.asarray(m)


    @property
    def turbine_positions(self):
        """The positions of turbines within the farm.
        :returns: The positions of turbines within the farm.
        :rtype: :func:`list`
        """
        return self.turbine_cache.cache["turbine_pos"]


    @property
    def turbine_frictions(self):
        """The friction coefficients of turbines within the farm.
        :returns: The friction coefficients of turbines within the farm.
        :rtype: :func:`list`
        """
        return self.turbine_cache.cache["turbine_friction"]


    def _update(self, m):
        """Updates the turbine cache."""
        self.turbine_cache(m)


    def add_turbine(self, coordinates, turbine=None):
        """Add a turbine to the farm at the given coordinates.

        Creates a new turbine of the same specification as the prototype turbine
        and places it at coordinates.

        :param coordinates: The x-y coordinates where the turbine should be placed.
        :type coordinates: :func:`list`
        :param turbine: A prototype turbine, see :doc:`opentidalfarm.turbine.Turbine`.
        """
        if self._turbine_prototype is None:
            if turbine is None:
                raise ValueError("A prototype turbine must be specified using "
                                 "the `turbine` keyword in "
                                 "<OpenTidalFarm.Farm.add_turbine>")
            else:
                self._turbine_prototype = turbine

        turbine = self._turbine_prototype

        self.turbine_cache.cache["turbine_friction"].append(turbine.friction)
        self.turbine_cache.cache["turbine_pos"].append(coordinates)
        dolfin.info("Turbine added at (%.2f, %.2f)." % (coordinates[0],
                                                        coordinates[1]))


    def _regular_turbine_layout(self, num_x, num_y, site_x_start, site_x_end,
                                site_y_start, site_y_end, turbine=None):
        """Adds a rectangular turbine layout to the farm.

        A rectangular turbine layout with turbines evenly spread out in each
        direction across the domain.

        :param turbine: Defines the type of turbine to add to the farm.
        :type turbine: Turbine object.
        :param num_x: The number of turbines placed in the x-direction.
        :type num_x: int
        :param num_y: The number of turbines placed in the y-direction.
        :type num_y: int
        :param site_x_start: The minimum x-coordinate of the site.
        :type site_x_start: float
        :param site_x_end: The maximum x-coordinate of the site.
        :type site_x_end: float
        :param site_y_start: The minimum y-coordinate of the site.
        :type site_y_start: float
        :param site_y_end: The maximum y-coordinate of the site.
        :type site_y_end: float
        :raises: ValueError

        """
        if self._turbine_prototype is None:
            if turbine is None:
                raise ValueError("A prototype turbine must be defined.")
            else:
                self._turbine_prototype = turbine

        turbine = self._turbine_prototype

        # Create an empty list of turbines.
        turbines = []
        # Generate the start and end points in the desired layout.
        start_x = site_x_start + turbine.radius
        start_y = site_y_start + turbine.radius
        end_x = site_x_end - turbine.radius
        end_y = site_y_end - turbine.radius
        # Check that we can fit enough turbines in each direction.
        too_many_x = turbine.diameter*num_x > end_x-start_x
        too_many_y = turbine.diameter*num_y > end_y-start_y
        # Raise exceptions if too many turbines are placed in a certain
        # direction.
        if too_many_x and too_many_y:
            raise ValueError("Too many turbines in the x and y direction")
        elif too_many_x:
            raise ValueError("Too many turbines in the x direction")
        elif too_many_y:
            raise ValueError("Too many turbines in the y direction")
        # Iterate over the x and y positions and append them to the turbine
        # list.
        else:
            for x in numpy.linspace(start_x, end_x, num_x):
                for y in numpy.linspace(start_y, end_y, num_y):
                    self.add_turbine((x,y), turbine)

            dolfin.info("Added %i turbines to the site in an %ix%i rectangular "
                        "array." % (num_x*num_y, num_x, num_y))


    def set_turbine_positions(self, positions):
        """Sets the turbine position and an equal friction parameter.

        :param list positions: List of tuples containint x-y coordinates of
            turbines to be added.
        """
        self.turbine_cache["turbine_pos"] = positions
        self.turbine_cache["turbine_friction"] = (
            self._turbine_prototype.friction*numpy.ones(len(positions)))


    def site_boundary_constraints(self):
        """Raises NotImplementedError if called."""
        return NotImplementedError("The Farm base class does not have "
                                   "boundaries.")


    def minimum_distance_constraints(self):
        """Returns an instance of MinimumDistanceConstraints.

        :returns: An instance of InequalityConstraint defining the minimum distance between turbines.
        :rtype: :doc:`opentidalfarm.farm.MinimumDistanceConstraints`

        """
        # Check we have some turbines.
        if self.number_of_turbines < 1:
            raise ValueError("Turbines must be deployed before minimum "
                             "distance constraints can be calculated.")

        m = self.as_parameter_array
        controls = self._turbine_prototype.controls
        minimum_distance = self._turbine_prototype.minimum_distance
        return MinimumDistanceConstraints(m, minimum_distance, controls)
