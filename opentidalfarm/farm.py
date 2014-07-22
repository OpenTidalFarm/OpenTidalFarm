"""
.. module:: Farm
   :synopsis: This modules provides Farm classes which contain information about
       the turbine farm.

"""

import numpy as np
import dolfin
from dolfin_adjoint import Constant
from minimum_distance_constraints import MinimumDistanceConstraints
from turbine import Turbine
from table import tabulate_data

class Farm(object):
    """A base Farm class from which other Farm classes should be derived."""
    def __init__(self):
        """Create the Farm."""
        self.turbines = []


    def only_position_constraints_enabled(self, serialized):
        return len(serialized) == 2*len(self.turbines)


    def only_friction_constraints_enabled(self, serialized):
        return len(serialized) == len(self.turbines)


    def friction_and_position_constraints_enabled(self, serialized):
        return len(serialized) == 3*len(self.turbines)


    def serialize(self, turbine_pos, turbine_friction):
        """Returns the serialized paramaterisation of the turbines.

        :param turbine_pos: Specifies whether or not turbine_pos is a control
            parameter.
        :type turbine_pos: Boolean
        :param turbine_friction: Specifies whether or not turbine_friction is a
            control parameter.
        :type turbine_friction: Boolean
        :returns: numpy.ndarray -- containing the serialized control parameters.

        """
        parameters = []
        if turbine_friction:
            for turbine in self.turbines:
                parameters.append(turbine.friction)
        if turbine_pos:
            for turbine in self.turbines:
                parameters.append(turbine.x)
                parameters.append(turbine.y)
        return np.array(parameters)


    def deserialize(self, serialized):
        """Deserializes the current parameter set.

        :param serialized: Serialized turbine paramaterisation of length 1, 2,
            or 3 times the number of turbines in the farm.
        :type serialized: numpy.ndarray.
        :raises: ValueError.

        """
        # Work out what we are updating
        both = len(serialized)==len(self.turbines)*3
        turbine_pos =  len(serialized)==len(self.turbines)*2
        turbine_friction = len(serialized)==len(self.turbines)

        if not both and not turbine_friction and not turbine_pos:
            raise ValueError("The parameter set is of the wrong length. It "
                             "should be of length 1, 2, or 3 times the number "
                             "of turbines.")

        # Turbine_friction and turbine_pos are control parameters
        if both:
            # Break up the input parameters into friction and tuples of
            # coordinates.
            friction = serialized[:len(serialized)/3]
            positions = serialized[len(serialized)/3:]
            positions = [(positions[2*i], positions[2*i+1])
                         for i in range(len(positions)/2)]
            # Set the value for each turbine.
            for i, d in enumerate(zip(friction, positions)):
                self.turbines[i].friction = d[0]
                self.turbines[i].coordinates = d[1]

        # Only turbine_friction is a control parameter.
        elif turbine_friction:
            for i, f in enumerate(serialized):
                self.turbines[i].friction = f
        # Only the turbine_pos is a control parameter.
        elif turbine_pos:
            for i in range(len(serialized)/2):
                position = (serialized[2*i], serialized[2*i+1])
                self.turbines[i].coordinates = position


    def add_turbine(self, turbine, coordinates):
        """Add a turbine to the farm at the given coordinates.

        Creates a new turbine of the same specification as turbine and places it
        at coordinates.

        :param turbine: A :py:class:`Turbine` class object describing the type
            of turbine to be added.
        :type turbine: :py:class:`Turbine`.
        :param coordinates: The x-y coordinates defining the location of the
            turbine to be added.
        :type coordinates: tuple of float.

        """
        # Add the turbine to the turbine list.
        self.turbines.append(turbine._copy_constructor(coordinates))


    def _regular_turbine_layout(self, turbine, num_x, num_y, site_x_start,
            site_x_end, site_y_start, site_y_end):
        """Generates a rectangular turbine layout.

        A rectangular turbine layout with turbines evenly spread out in each
        direction across the domain.

        Args:
            num_x: An int with the number of turbines placed in the x-direction.
            num_y: An int with the number of turbines placed in the y-direction.
            site_x_start: A float defining the minimum x-coordinate of the site
                where turbines are to be placed.
            site_x_end: A float defining the minimum x-coordinate of the site
                where turbines are to be placed.
            site_y_start: A float defining the minimum y-coordinate of the site
                where turbines are to be placed.
            site_y_end: A float defining the minimum y-coordinate of the site
                where turbines are to be placed.
        """
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
            for x in np.linspace(start_x, end_x, num_x):
                for y in np.linspace(start_y, end_y, num_y):
                    self.add_turbine(turbine, (x,y))
                    dolfin.log(dolfin.PROGRESS,
                               "Added a turbine at (%.3f, " "%.3f)." % (x, y))

            dolfin.log(dolfin.INFO,
                       "Added %i turbines to the site in an %ix%i rectangular "
                       "array." % (num_x*num_y, num_x, num_y))


    def site_boundary_constraints(self):
        """Raises NotImplementedError if called."""
        return NotImplementedError("The Farm base class does not have "
                                   "boundaries.")


# TODO: testing and documentation
# TODO: how does the farm know about the control parameters?

    def minimum_distance_constraints(self, turbine_pos, turbine_friction):
        """Returns an instance of MinimumDistanceConstraints."""
        # Check we have some turbines.
        if len(self.turbines) == 0:
            raise NotImplementedError("Turbines must be deployed before "
                                      "minimum distance constraints can be "
                                      "calculated.")

        m = self.serialize(turbine_pos, turbine_friction)
        minimum_distance = self.turbines[0].minimum_distance
        friction_and_position_enabled = \
                               self.friction_and_position_constraints_enabled(m)
        return MinimumDistanceConstraints(m, minimum_distance,
                                          friction_and_position_enabled)



class RectangularFarm(Farm):
    def __init__(self, site_x_start, site_x_end, site_y_start, site_y_end):
        # Initialize the base class
        super(RectangularFarm, self).__init__()

        self._site_x_start = site_x_start
        self._site_y_start = site_y_start
        self._site_x_end = site_x_end
        self._site_y_end = site_y_end


    def __str__(self):
        header = "Farm information"
        data = [("Number of turbines", len(self.turbines)),
                ("Site limits (x) / m",
                 str(self.site_x_start)+"--"+str(self.site_x_end)),
                ("Site limits (y) / m",
                 str(self.site_y_start)+"--"+str(self.site_y_end))]
        return tabulate_data(header, data)


    @property
    def site_x_start(self):
        """The smallest x-coordinate of the site"""
        return self._site_x_start


    @property
    def site_y_start(self):
        """The smallest y-coordinate of the site"""
        return self._site_y_start


    @property
    def site_x_end(self):
        """The largest x-coordinate of the site"""
        return self._site_x_end


    @property
    def site_y_end(self):
        """The largest y-coordinate of the site"""
        return self._site_y_end


    def add_regular_turbine_layout(self, turbine, num_x, num_y, x_start=None,
                                   x_end=None, y_start=None, y_end=None):
        # Get the documentation from the base function
        self.__doc__ = \
                super(RectangularFarm, self)._regular_turbine_layout.__doc__

        # Get default parameters:
        if x_start is None: x_start = self.site_x_start
        if y_start is None: y_start = self.site_y_start
        if x_end is None: x_end = self.site_x_end
        if y_end is None: y_end = self.site_y_end

        return super(RectangularFarm, self)._regular_turbine_layout(turbine,
                                                                    num_x,
                                                                    num_y,
                                                                    x_start,
                                                                    x_end,
                                                                    y_start,
                                                                    y_end)


    def site_boundary_constraints(self):
        """Returns the site bounrady constraints for a rectangular site.

        These constraints ensure that the turbine positions remain within the
        turbine site during optimisation.

        Returns:
            A tuple of lists of length equal to the twice the number of turbines.
            Each list contains dolfin_adjoint.Constant objects of the upper and
            lower bound coordinates.

        Raises:
            ValueError: If there are no turbines deployed.
            ValueError: If the lower bounds are larger than the upper bounds.
        """
        n_turbines = len(self.turbines)
        # Check we have deployed some turbines in the farm.
        if n_turbines == 0:
            raise ValueError("You must deploy turbines before computing "
                             "position constraints.")

        largest_radius = max([t.radius for t in self.turbines])
        # Get the lower and upper bounds.
        lower_x = self.site_x_start + largest_radius
        lower_y = self.site_y_start + largest_radius
        upper_x = self.site_x_end - largest_radius
        upper_y = self.site_y_end - largest_radius

        # Check the site is large enough.
        if upper_x < lower_x or upper_y < lower_y:
            raise ValueError("Lower bound is larger than upper bound. Is your "
                             "domain large enough?")

        # The control variable is ordered as [t1_x, t1_y, t2_x, t2_y, t3_x, ...]
        lower_bounds = n_turbines*[Constant(lower_x), Constant(lower_y)]
        upper_bounds = n_turbines*[Constant(upper_x), Constant(upper_y)]
        return lower_bounds, upper_bounds
