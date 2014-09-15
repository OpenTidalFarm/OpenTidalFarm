import os.path
import numpy as np
import dolfin
from dolfin_adjoint import Constant
from .minimum_distance_constraints import MinimumDistanceConstraints
from ..turbine import Turbine

class BaseFarm(object):
    """A base Farm class from which other Farm classes should be derived."""
    def __init__(self):
        """Create an empty Farm."""
        self._turbine_prototype = None
        self._turbine_parameterisation = None
        self._turbine_positions = []
        self._friction = []
        self._number_of_turbines = 0


    @property
    def number_of_turbines(self):
        return self._number_of_turbines


    def serialize(self, controls):
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

        # TODO: Dynamic friction
        if controls.friction:
            for i in xrange(self.number_of_turbines):
                parameters.append(self._friction[i])
        if controls.position:
            for i in xrange(self.number_of_turbines):
                parameters.append(self._turbine_positions[2*i])
                parameters.append(self._turbine_positions[2*i+1])

        return np.asarray(parameters)


    def deserialize(self, serialized, controls):
        """Deserializes the current parameter set.

        :param serialized: Serialized turbine paramaterisation of length 1, 2,
            or 3 times the number of turbines in the farm.
        :type serialized: numpy.ndarray.
        :raises: ValueError.

        """
        # Turbine_friction and turbine_pos are control parameters
        if controls.friction and controls.position:
            # Break up the input parameters into friction and tuples of
            # coordinates.
            self._friction = serialized[:len(serialized)/3]
            self._turbine_positions = serialized[len(serialized)/3:]

        # Only turbine_friction is a control parameter.
        elif controls.friction:
            self._friction = serialized
        # Only the turbine_pos is a control parameter.
        elif controls.position:
            self._turbine_positions = serialized


    def add_turbine(self, coordinates, turbine=None):
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
        if self._turbine_prototype is None:
            if turbine is None:
                raise ValueError("A prototype turbine must be specified using "
                                 "the `turbine` keyword in "
                                 "<OpenTidalFarm.Farm.add_turbine>")
            else:
                self._turbine_prototype = turbine

        elif turbine is None:
            turbine = self._turbine_prototype

        self._friction.append(turbine.friction)
        self._turbine_positions.append(coordinates[0])
        self._turbine_positions.append(coordinates[1])
        self._number_of_turbines += 1
        dolfin.info("Turbine added at (%.2f, %.2f). %i turbines within the "
                    "farm." % (coordinates[0], coordinates[1],
                               self.number_of_turbines))


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
        elif self._turbine_prototype!=turbine:
            raise ValueError("A prototype turbine has already been defined!")

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
            for x in np.linspace(start_x, end_x, num_x):
                for y in np.linspace(start_y, end_y, num_y):
                    self.add_turbine((x,y), turbine)

            dolfin.info("Added %i turbines to the site in an %ix%i rectangular "
                        "array." % (num_x*num_y, num_x, num_y))


    def site_boundary_constraints(self):
        """Raises NotImplementedError if called."""
        return NotImplementedError("The Farm base class does not have "
                                   "boundaries.")


    def minimum_distance_constraints(self, controls):
        """Returns an instance of MinimumDistanceConstraints.

        :param controls: The optimisation controls.
        :type controls: Controls
        :raises: ValueError
        :returns: InequalityConstraint instance defining the minimum distance
            between turbines.

        """
        # Check we have some turbines.
        if self.number_of_turbines < 1:
            raise ValueError("Turbines must be deployed before minimum "
                             "distance constraints can be calculated.")

        m = self.serialize(controls)
        minimum_distance = self._turbine_prototype.minimum_distance
        return MinimumDistanceConstraints(m, minimum_distance, self._controls)
