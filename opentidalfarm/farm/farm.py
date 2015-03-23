import os
from dolfin import FunctionSpace
from .base_farm import BaseFarm


class Farm(BaseFarm):
    """Extends :py:class:`BaseFarm`. Creates a farm from a mesh and subdomain ids.

    Following parameters are available:

    :ivar domain: A :class:`Domain` object describing the domain.
    :ivar turbine: A :class:`Turbine` object describing the domain.
    :ivar site_ids: A list of integers describing the subdomain identifiers of
        the farm(s)
    :ivar function_space: A :class:`dolfin.FunctionSpace` that specifies in
        which function space the turbine friction is in.

    """
    def __init__(self, domain, turbine=None, site_ids=None, function_space=None):
        # Initialize the base class
        super(Farm, self).__init__(domain, turbine, site_ids)

        if function_space is None:
            function_space = FunctionSpace(self.domain.mesh, "CG", 2)
        self._turbine_function_space = function_space

        # Set the function space in the cache.
        self.turbine_cache.set_function_space(function_space)
