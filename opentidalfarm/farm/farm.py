import os
from dolfin import FunctionSpace
from .base_farm import BaseFarm


class Farm(BaseFarm):
    """Extends :py:class:`BaseFarm`. Creates a farm from a mesh.

    This class holds the turbines within a site defined by a mesh with a turbine
    site marked by `1`.

    """
    def __init__(self, domain):
        """Initializes an empty farm defined by the domain.

        """
        # Initialize the base class
        super(Farm, self).__init__()

        # Store the domain.
        self.domain = domain

        # Create a turbine function space and set the function space in the
        # cache.
        self._turbine_function_space = FunctionSpace(self.domain.mesh, "CG", 2)
        self.turbine_cache._function_space = self._turbine_function_space
