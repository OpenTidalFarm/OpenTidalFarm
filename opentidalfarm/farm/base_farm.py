import numpy
import dolfin
from ..optimisation_helpers import MinimumDistanceConstraints
from ..turbine_cache import TurbineCache

class BaseFarm(object):
    """A base Farm class from which other Farm classes should be derived."""
    def __init__(self, domain=None, turbine=None, site_ids=None):
        """Create an empty Farm."""
        # Create a chaching object for the interpolated turbine friction fields
        # (as their computation is very expensive)
        self.turbine_cache = TurbineCache()
        self._parameters = {"friction": [], "position": []}

        self.domain = domain
        self._set_turbine_specification(turbine)

        # The measure of the farm site
        self.site_dx = self.domain.dx(site_ids)

    def update(self):
        self.turbine_cache.update(self)

    @property
    def friction_function(self):
        self.update()
        return self.turbine_cache["turbine_field"]


    def _get_turbine_specification(self):
        if self._turbine_specification is None:
            raise ValueError("The turbine specification has not yet been set.")
        return self._turbine_specification


    def _set_turbine_specification(self, turbine_specification):
        self._turbine_specification = turbine_specification
        self.turbine_cache.set_turbine_specification(
            self._turbine_specification)


    turbine_specification = property(_get_turbine_specification,
                                     _set_turbine_specification,
                                     "The turbine specification.")

    @property
    def number_of_turbines(self):
        """The number of turbines in the farm.
        :returns: The number of turbines in the farm.
        :rtype: int
        """
        return len(self.turbine_cache["position"])


    @property
    def control_array(self):
        """A serialized representation of the farm based on the controls.

        :returns: A serialized representation of the farm based on the controls.
        :rtype: numpy.ndarray
        """

        if self._turbine_specification.smeared:
            return self.friction_function.vector().array()

        else:
            m = []

            if (self._turbine_specification.controls.friction or
                self._turbine_specification.controls.dynamic_friction):
                m += numpy.reshape(
                    self._parameters["friction"], -1).tolist()

            if self._turbine_specification.controls.position:
                m += numpy.reshape(
                    self._parameters["position"], -1).tolist()

            return numpy.asarray(m)


    @property
    def turbine_positions(self):
        """The positions of turbines within the farm.
        :returns: The positions of turbines within the farm.
        :rtype: :func:`list`
        """
        return self._parameters["position"]


    @property
    def turbine_frictions(self):
        """The friction coefficients of turbines within the farm.
        :returns: The friction coefficients of turbines within the farm.
        :rtype: :func:`list`
        """
        return self._parameters["friction"]


    def add_turbine(self, coordinates):
        """Add a turbine to the farm at the given coordinates.

        Creates a new turbine of the same specification as the prototype turbine
        and places it at coordinates.

        :param coordinates: The x-y coordinates where the turbine should be placed.
        :type coordinates: :func:`list`
        """
        if self._turbine_specification is None:
            raise ValueError("A turbine specification has not been set.")

        turbine = self._turbine_specification
        self._parameters["position"].append(coordinates)
        self._parameters["friction"].append(turbine.friction)

        dolfin.info("Turbine added at (%.2f, %.2f)." % (coordinates[0],
                                                        coordinates[1]))

    def _staggered_turbine_layout(self, num_x, num_y, site_x_start, site_x_end,
                                site_y_start, site_y_end):
        """Adds a staggered, rectangular turbine layout to the farm.

        A rectangular turbine layout with turbines evenly spread out in each
        direction across the domain.

        :param turbine: Defines the type of turbine to add to the farm.
        :type turbine: Turbine object.
        :param num_x: The number of turbines placed in the x-direction.
        :type num_x: int
        :param num_y: The number of turbines placed in the y-direction (will be one less in each second row).
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
        if self._turbine_specification is None:
            raise ValueError("A turbine specification has not been set.")

        turbine = self._turbine_specification

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
        for i, x in enumerate(numpy.linspace(start_x, end_x, num_x)):
            if i % 2 == 0:
                for y in numpy.linspace(start_y, end_y, num_y):
                    self.add_turbine((x,y))
            else:
                ys = numpy.linspace(start_y, end_y, num_y)
                for i in range(len(ys)-1):
                    self.add_turbine((x, ys[i] + 0.5*(ys[i+1]-ys[i])))

        dolfin.info("Added %i turbines to the site in an %ix%i rectangular "
                    "array." % (num_x*num_y, num_x, num_y))

    def _regular_turbine_layout(self, num_x, num_y, site_x_start, site_x_end,
                                site_y_start, site_y_end):
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
        if self._turbine_specification is None:
            raise ValueError("A turbine specification has not been set.")

        turbine = self._turbine_specification

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
        for x in numpy.linspace(start_x, end_x, num_x):
            for y in numpy.linspace(start_y, end_y, num_y):
                self.add_turbine((x,y))

        dolfin.info("Added %i turbines to the site in an %ix%i rectangular "
                    "array." % (num_x*num_y, num_x, num_y))


    def set_turbine_positions(self, positions):
        """Sets the turbine position and an equal friction parameter.

        :param list positions: List of tuples containint x-y coordinates of
            turbines to be added.
        """
        self.turbine_cache["position"] = positions
        self.turbine_cache["friction"] = (
            self._turbine_specification.friction*numpy.ones(len(positions)))
        self.update()


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
        n_turbines = len(self.turbine_positions)
        if (n_turbines < 1):
            raise ValueError("Turbines must be deployed before minimum "
                             "distance constraints can be calculated.")

        controls = self._turbine_specification.controls
        minimum_distance = self._turbine_specification.minimum_distance
        positions = self.turbine_positions
        return MinimumDistanceConstraints(self, positions, minimum_distance, controls)
