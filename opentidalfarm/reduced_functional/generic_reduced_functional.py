import numpy as np
from dolfin import *
from dolfin_adjoint import *
import dolfin_adjoint


class GenericReducedFunctional(dolfin_adjoint.ReducedFunctionalNumPy):
    """Generic reduced functional object 
    
    This should be overloaded by implemented reduced functionals, this ensures
    that reduced functional objects for different solvers may be scaled and
    combined, and that the requisite methods are present in order to
    interface with the dolfin-adjoint optimsation framework
    
    .. note::
        
        __init__, reduced_functional and derivative must be overloaded 

    """
    
    def __init__(self):
        raise NotImplementedError('GenericReducedFunctional.__init__ needs \
                to be overloaded')        
    
    def __call__(self, m):
        """ Interface function for dolfin_adjoint.ReducedFunctional, this 
        method does not require overloading, it redirects to the \
        reduced_functional method to preserve naming consitency. It then \
        returns the functional value for the parameter choice

        :param m: cartesian co-ordinates of the turbines
        :type m: TODO
         bool
        """
        return self.reduced_functional(m)

    def reduced_functional(self, m):
        """ This should be overloaded and should return the functional value \
        for the parameter choice

        :param m: cartesian co-ordinates of the turbines
        :type m: TODO
         bool
        """
        raise NotImplementedError('GenericReducedFunctional.reduced_functional \
                needs to be overloaded')

    def derivative(self, m_array, taylor_test=False, seed=0.001, forget = \
            True, **kwargs):
        """ Interface function for dolfin_adjoint.ReducedFunctional, this 
        method should return the derivative of the functional value with 
        respect to the parameter choice
        
        :param m_array: cartesian co-ordinates of the turbines.
        :type m_array: 2n x 1 numpy array.
        :param taylor_test: setting to check consistency of derivative returned.
        :type taylor_test: bool.
        :param seed: TODO.
        :type seed: float (in km).
        :param forget: TODO.
        :type forget: bool.
        
        """
        raise NotImplementedError('GenericReducedFunctional.derivative needs \
                to be overloaded')

    def hessian(self, m_array, m_dot_array):
        """ Interface function for dolfin_adjoint.ReducedFunctional, this 
        method should return the Hessian of the functional value with 
        respect to the parameter choices
        
        :param m_array: cartesian co-ordinates of the turbines.
        :type m_array: 2n x 1 numpy array.
        :param m_dot_array: TODO.
        :type m_dot_array: TODO.        
        
        """
        raise NotImplementedError('GenericReducedFunctional.hessian needs \
                to be overloaded')
                
    def __add__(self, other):
        """ Method to add reduced functionals together"""
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
        
        
class CombinedReducedFunctional(GenericReducedFunctional):
    """ Constructs a single combined functional by adding one functional to 
    another.
    """
    
    def __init__(self, reduced_functional_list):
        for reducedfunctional in reduced_functional_list:
            assert isinstance(reducedfunctional, dolfin_adjoint.ReducedFunctionalNumPy) 
        self.reduced_functional_list = reduced_functional_list
        
    def __call__(self, m):
        """Return the functional value for the parameter choice"""
        combined_reduced_functional = sum([reducedfunctional.__call__(m) for \
                reducedfunctional in self.reduced_functional_list])
        return combined_reduced_functional
        
    def derivative(self, m_array, taylor_test=False, seed=0.001, forget =   \
            True, **kwargs):
        """ Return the derivative of the functional value with respect to 
        the parameter choice"""
        combined_reduced_functional_derivative = \
                sum([reducedfunctional.derivative(m_array, taylor_test =    \
                False, seed=0.001, forget = True, **kwargs) for             \
                reducedfunctional in self.reduced_functional_list])
        return combined_reduced_functional_derivative
        
    def hessian(self, m_array, m_dot_array):
        """ Hessian not implemented """
        raise NotImplementedError('The Hessian computation is not yet \
                implemented')
        
        
class ScaledReducedFunctional(GenericReducedFunctional):
    """Scales the functional 
    """
    10, 
    def __init__(self, reducedfunctional, scaling_factor):
        assert isinstance(reducedfunctional, dolfin_adjoint.ReducedFunctionalNumPy) 
        assert isinstance(scaling_factor, int) or isinstance(scaling_factor, float)
        self.reducedfunctional = reducedfunctional
        self.scaling_factor = scaling_factor  
        
    def __call__(self, m):
        """Return the functional value for the parameter choice"""
        scaled_reduced_functional = self.scaling_factor *                   \
                self.reducedfunctional.__call__(m)
        return scaled_reduced_functional
        
    def derivative(self, m_array, taylor_test=False, seed=0.001, forget =   \
            True, **kwargs):
        """ Return the derivative of the functional value with respect to 
        the parameter choice"""
        scaled_reduced_functional_derivative = self.scaling_factor *        \
                self.reducedfunctional.derivative(m_array, taylor_test =    \
                False, seed=0.001, forget = True, **kwargs)
        return scaled_reduced_functional_derivative
        
    def hessian(self, m_array, m_dot_array):
        """ Hessian not implemented """
        raise NotImplementedError('The Hessian computation is not yet \
                implemented') 


#############################################################################
################################## T E S T ##################################
#############################################################################        
        
        
if __name__ == '__main__':
    
    class Test(GenericReducedFunctional):
        
        def __init__(self):
            print 'Initialised...'
        
        def reduced_functional(self, m):
            print 'Running reduced_functional method'
            return sum(m)
        
        def derivative(self, m_array, taylor_test=False, seed=0.001, forget = \
                True):
            print 'Running derivative method'
            return m_array
            
        def hessian(self, m_array, m_dot_array):
            print 'n/a'

    TestA = Test()
    TestB = Test()
    objective = (2 * TestA) + TestB
    print objective(np.array([1,2,3]))
    print objective.derivative(np.array([1,2,3]))
