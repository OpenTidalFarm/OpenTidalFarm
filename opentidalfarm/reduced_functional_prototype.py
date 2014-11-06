from dolfin_adjoint import ReducedFunctionalNumPy


class ReducedFunctionalPrototype(ReducedFunctionalNumPy):
    """Generic reduced functional object

    This should be overloaded by implemented reduced functionals, this ensures
    that reduced functional objects for different solvers may be scaled and
    combined, and that the requisite methods are present in order to
    interface with the dolfin-adjoint optimsation framework

    .. note::

        reduced_functional and derivative must be overloaded

    """

    def __init__(self):
        pass

    def __call__(self, m, **kwargs):
        """ Interface function for dolfin_adjoint.ReducedFunctional, this
        method does not require overloading, it redirects to the \
        reduced_functional method to preserve naming consitency. It then \
        returns the functional value for the control choice

        :param m: The control values
        :type m: numpy array
        """
        return self.evaluate(m, **kwargs)

    def evaluate(self, m, **kwargs):
        """ This should be overloaded and should return the functional value \
        for the control choice

        :param m: The control values
        :type m: numpy array
        """
        raise NotImplementedError('ReducedFunctionalPrototype.reduced_functional \
                needs to be overloaded')

    def derivative(self, m, **kwargs):
        """ Interface function for dolfin_adjoint.ReducedFunctional, this
        method should return the derivative of the functional value with
        respect to the control choice

        :param m: The control values
        :type m: numpy array.
        """
        raise NotImplementedError('ReducedFunctionalPrototype.derivative needs \
                to be overloaded')

    def __add__(self, other):
        """ Method to add reduced functionals together"""
        assert self.controls == other.controls
        return CombinedReducedFunctional([self, other])

    def __sub__(self, other):
        """ method to subtract one reduced functional from another """
        return CombinedReducedFunctional([self, -other])

    def __mul__(self, other):
        """ method to scale a reduced functional """
        return ScaledReducedFunctional(self, other)

    def __rmul__(self, other):
        """ preserves commutativity of scaling """
        return ScaledReducedFunctional(self, other)

    def __neg__(self):
        """ implements the negative of the reduced functional """
        return -1 * self


class CombinedReducedFunctional(ReducedFunctionalPrototype):
    """ Constructs a single combined functional by adding one functional to
    another.
    """

    def __init__(self, reduced_functional_list):
        for reducedfunctional in reduced_functional_list:
            assert isinstance(reducedfunctional, ReducedFunctionalNumPy)
        self.reduced_functional_list = reduced_functional_list

		# We know that all controls are the same, so just pick the first one
        self.controls = reduced_functional_list[0].controls

    def __call__(self, m):
        """Return the functional value for the controls choice"""
        combined_reduced_functional = sum([reducedfunctional.__call__(m) for \
                reducedfunctional in self.reduced_functional_list])
        return combined_reduced_functional

    def derivative(self, m, **kwargs):
        """ Return the derivative of the functional value with respect to
        the control choice"""
        combined_reduced_functional_derivative = \
                sum([reducedfunctional.derivative(m, **kwargs) for \
                reducedfunctional in self.reduced_functional_list])
        return combined_reduced_functional_derivative


class ScaledReducedFunctional(ReducedFunctionalPrototype):
    """Scales the functional
    """
    def __init__(self, reducedfunctional, scaling_factor):
        assert isinstance(reducedfunctional, ReducedFunctionalNumPy)
        assert isinstance(scaling_factor, int) or isinstance(scaling_factor, float)
        self.reducedfunctional = reducedfunctional
        self.scaling_factor = scaling_factor
        self.controls = reducedfunctional.controls

    def __call__(self, m):
        """Return the functional value for the controls choice"""

        scaled_reduced_functional = self.scaling_factor *                   \
                self.reducedfunctional(m)
        return scaled_reduced_functional

    def derivative(self, m, **kwargs):
        """ Return the derivative of the functional value with respect to
        the control choice"""
        scaled_reduced_functional_derivative = self.scaling_factor *        \
                self.reducedfunctional.derivative(m, **kwargs)
        return scaled_reduced_functional_derivative


#############################################################################
################################## T E S T ##################################
#############################################################################

class TestReducedFunctional(ReducedFunctionalPrototype):

    def __init__(self, controls):
        if not hasattr(controls, "__getitem__"):
            controls = [controls]
        self.controls = controls

    def evaluate(self, m, **kwargs):
        return sum(m)

    def derivative(self, m, **kwargs):
        return m

if __name__ == '__main__':
    import numpy as np

    class Test(ReducedFunctionalPrototype):
        def __init__(self):
            self.controls = []

        def evaluate(self, m):
            print 'Running reduced_functional method'
            return sum(m)

        def derivative(self, m):
            print 'Running derivative method'
            return m

    TestA = Test()
    TestB = Test()
    objective = (2 * TestA) + TestB
    print objective(np.array([1,2,3]))
    print objective.derivative(np.array([1,2,3]))
