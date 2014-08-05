import numpy
from generic_reduced_functional import GenericReducedFunctional

class ShallowWaterReducedFunctional(GenericReducedFunctional):
    """Constructs the reduced functional object for the shallow water solver

    Overloads the GenericReducedFunctional

    .. note::
        
        The key dolfin-adjoint interface methods are __init__, __call__
        derivative and hessian.

    """


    def __init__(self, farm, functional, turbine_position=False,
            turbine_friction=False):
        """Inits ShallowWaterReducedFunctional class with:

        :param farm: TODO
        :type farm: TODO
        :param functional: The objective functional of the shallow water solver
        :type functional: GenericFunctional class object
        :param control: The control parameter (turbine pos / friction)
        :type 

        """
        self.farm = farm
        self.functional = functional
        
        #Set up the memoization wrappers
        self.compute_functional_mem = memoize.MemoizeMutable(compute_functional,
                                      hash_keys)
        self.compute_gradient_mem = memoize.MemoizeMutable(compute_gradient,
                                    hash_keys)
        self.compute_hessian_action_mem =
            memoize.MemoizeMutable(compute_hessian_action, hash_keys)

        self._turbine_position = turbine_position
        self._turbine_friction = turbine_friction


        m = self.farm.deserialize(self._turbine_position,
                self._turbine_friction)

    def reduced_functional(self, m):
        """returns the reduced functional from the turbine locations

        :param m: cartesian co-ordinates of the turbines.
        :type m: TODO.
        
        .. note::

            This is method is the top of many layers; m is passed in and it is
            converted to a friction function from which the functional is
            calculated - these steps are memoized. The functional is then scaled
            as required before being returned here.
        """
        return self._scaled_reduced_functional(m)

    def _scaled_reduced_functional(self, m)

    def derivative(self, m_arrya
