from dolfin import FunctionSpace
import dolfin_adjoint
from .farm import Farm

class RectangularFarm(Farm):
    """Extends :py:class:`Farm`. Defines a rectangular Farm.

    This class holds the turbines within a rectangular site.

    """
    def __init__(self, domain, site_x_start, site_x_end, site_y_start,
                 site_y_end, turbine=None, site_ids=None):
        """Initializes an empty rectangular farm with the given dimensions.

        :param mesh: The name of the mesh file to use, e.g. 'mesh.xml' if the
            file is located at `./mesh.xml`.
        :type mesh: string
        :param site_x_start: The minimum x-coordinate for the site.
        :type site_x_start: float
        :param site_x_end: The maximum x-coordinate for the site.
        :type site_x_end: float
        :param site_y_start: The minimum y-coordinate for the site.
        :type site_y_start: float
        :param site_y_end: The maximum y-coordinate for the site.
        :type site_y_end: float

        """
        # Initialize the base clas
        super(RectangularFarm, self).__init__(domain, turbine, site_ids)

        # Create a turbine function space and set the function space in the
        # cache.
        self._turbine_function_space = FunctionSpace(self.domain.mesh, "CG", 2)
        self.turbine_cache.set_function_space(self._turbine_function_space)

        # Store site dimensions.
        self._site_x_start = site_x_start
        self._site_y_start = site_y_start
        self._site_x_end = site_x_end
        self._site_y_end = site_y_end


    @property
    def site_x_start(self):
        """The minimum x-coordinate of the site.

        :getter: Returns the minimum x-coordinate of the site.
        :type: float
        """
        return self._site_x_start


    @property
    def site_y_start(self):
        """The minimum y-coordinate of the site.

        :getter: Returns the minimum y-coordinate of the site.
        :type: float
        """
        return self._site_y_start


    @property
    def site_x_end(self):
        """The maximum x-coordinate of the site.

        :getter: Returns the maximum x-coordinate of the site.
        :type: float
        """
        return self._site_x_end


    @property
    def site_y_end(self):
        """The maximum y-coordinate of the site.

        :getter: Returns the maximum y-coordinate of the site.
        :type: float
        """
        return self._site_y_end


    def add_regular_turbine_layout(self, num_x, num_y, x_start=None,
                                   x_end=None, y_start=None, y_end=None):
        """Adds a rectangular turbine layout to the farm.

        A rectangular turbine layout with turbines evenly spread out in each
        direction across the given rectangular site.

        :param turbine: Defines the type of turbine to add to the farm.
        :type turbine: Turbine object.
        :param num_x: The number of turbines placed in the x-direction.
        :type num_x: int
        :param num_y: The number of turbines placed in the y-direction.
        :type num_y: int
        :param x_start: The minimum x-coordinate of the site.
        :type x_start: float
        :param x_end: The maximum x-coordinate of the site.
        :type x_end: float
        :param y_start: The minimum y-coordinate of the site.
        :type y_start: float
        :param y_end: The maximum y-coordinate of the site.
        :type y_end: float
        :raises: ValueError

        """
        # Get default parameters:
        if x_start is None: x_start = self.site_x_start
        if y_start is None: y_start = self.site_y_start
        if x_end is None: x_end = self.site_x_end
        if y_end is None: y_end = self.site_y_end

        return super(RectangularFarm, self)._regular_turbine_layout(
            num_x, num_y, x_start, x_end, y_start, y_end)


    def add_staggered_turbine_layout(self, num_x, num_y, x_start=None,
                                   x_end=None, y_start=None, y_end=None):
        """Adds a rectangular, staggered turbine layout to the farm.

        A rectangular turbine layout with turbines evenly spread out in each
        direction across the given rectangular site.

        :param turbine: Defines the type of turbine to add to the farm.
        :type turbine: Turbine object.
        :param num_x: The number of turbines placed in the x-direction.
        :type num_x: int
        :param num_y: The number of turbines placed in the y-direction (will be one less on every second row).
        :type num_y: int
        :param x_start: The minimum x-coordinate of the site.
        :type x_start: float
        :param x_end: The maximum x-coordinate of the site.
        :type x_end: float
        :param y_start: The minimum y-coordinate of the site.
        :type y_start: float
        :param y_end: The maximum y-coordinate of the site.
        :type y_end: float
        :raises: ValueError

        """
        # Get default parameters:
        if x_start is None: x_start = self.site_x_start
        if y_start is None: y_start = self.site_y_start
        if x_end is None: x_end = self.site_x_end
        if y_end is None: y_end = self.site_y_end

        return super(RectangularFarm, self)._staggered_turbine_layout(
            num_x, num_y, x_start, x_end, y_start, y_end)


    def site_boundary_constraints(self):
        """Returns the site boundary constraints for a rectangular site.

        These constraints ensure that the turbine positions remain within the
        turbine site during optimisation.

        :raises: ValueError
        :returns: Tuple of lists of length equal to the twice the number of
            turbines. Each list contains dolfin_adjoint.Constant objects of the
            upper and lower bound coordinates.

        """
        # Check we have deployed some turbines in the farm.
        n_turbines = len(self.turbine_positions)
        if (n_turbines < 1):
            raise ValueError("You must deploy turbines before computing "
                             "position constraints.")

        radius = self._turbine_specification.radius
        # Get the lower and upper bounds.
        lower_x = self.site_x_start+radius
        lower_y = self.site_y_start+radius
        upper_x = self.site_x_end-radius
        upper_y = self.site_y_end-radius

        # Check the site is large enough.
        if upper_x < lower_x or upper_y < lower_y:
            raise ValueError("Lower bound is larger than upper bound. Is your "
                             "domain large enough?")

        # The control variable is ordered as [t1_x, t1_y, t2_x, t2_y, t3_x, ...]
        lower_bounds = n_turbines*[dolfin_adjoint.Constant(lower_x),
                                   dolfin_adjoint.Constant(lower_y)]
        upper_bounds = n_turbines*[dolfin_adjoint.Constant(upper_x),
                                   dolfin_adjoint.Constant(upper_y)]
        return lower_bounds, upper_bounds


