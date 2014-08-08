"""
.. module:: ReducedFunctionals
    :synopsis: Defines the various ReducedFunctional classes ready for being
    combined in the MasterReducedFunctional class.

"""

from generic_reduced_functional import GenericReducedFunctional
from shallow_water_reduced_functional import SWReducedFunctional,SWDerivative,
SWHessian, SWHelpers, SWAdditionalProperties

class ShallowWaterReducedFunctional(GenericReducedFunctional,
                                    SWReducedFunctional, SWDerivative, SWHessian,
                                    SWHelpers, SWAdditionalProperties):
    """Constructs the reduced functional object for the shallow water solver

    Overloads the GenericReducedFunctional

    .. note::

        The methods which must be overloaded are __init__, reduced_functional
        derivative and hessian.

    """

    def __init__(self, farm, objective):
        """Inits ShallowWaterReducedFunctional class with:

        :param farm: TODO
        :type farm: TODO
        :param functional: The objective functional of the shallow water solver
        :type functional: GenericFunctional class object
        :param control: The control parameter (turbine pos / friction)
        :type

        """
        # Make some things a little more accessible for convenience
        self.farm = farm
        self.functional = functional
        self._controls = objective._controls
        self._sw_problem = objective._solver._problem
        self._sw_solver = objective._solver
        self._parametrisation = farm._prototype_turbine._parametrisation
        self._turbine_cache = farm._turbine_cache
        self.function_space = self._sw_problem._function_space
        m = self.farm.deserialize(self._controls)

        # Set up the memoization wrappers
        self._memoized_reduced_functional = \
            memoize.MemoizeMutable(_reduced_functional_from_m, hash_keys)
        self._memoized_derivative = \
            memoize.MemoizeMutable(_derivative_from_m, hash_keys)
        self._memoized_hessian = \
            memoize.MemoizeMutable(_compute_hessian_action, hash_keys)

        # Set the defaults of some of the additional class parameters
        self._automatic_scaling = 1
        self._output_turbine_power = True
        self._checkpoint = False

        # Initialise some caching variables
        self.last_m = None
        self.last_state = None

        # Do some preparatory work
        if self._output_turbine_power:
            self.power_file = File(config.params["base_path"] + os.path.sep +
                    "power.pvd", "compressed")

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

    def derivative(self, m_array, taylor_test=False, seed=0.001, forget=True,
            **kwargs):
        """returns the functional derivative from the turbine locations and
        can perform a Taylor test

        :param m_array: cartesian co-ordinates of the turbines.
        :type m_array: TODO.
        :param taylor_test: Performs a Taylor test to check the gradient
        :type taylor_test: Boolean
        :param seed: seed the Taylor test
        :type seed: float
        :param forget: !!!TODO!!!
        :type forget: Boolean

        .. note::

            Like the reduced functional, this method is the top of many layers.
        """
        if taylor_test:
            return self._functional_derivative_with_check(m_array, seed, forget)
        else:
            return self._scaled_derivative(m_array, forget)

    def hessian(self, m, m_dot):
        """ Returns the hessian of the functional with respect to the control
        variable.

        :param m:
        :type m:
        :param m_dot:
        :type m_dot:

        .. note::
            not yet implemented.
        """
        raise NotImplementedError('The Hessian computation is not yet
                implemented')






































