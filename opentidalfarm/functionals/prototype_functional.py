"""
..module:: Prototype Functional
  :synopsis: This module provides the prototype functional class, which should
  be overwritten as required.
"""

class PrototypeFunctional(object):
    r"""Prototype functional object which should be overloaded by implemented
    functionals. '__init__' and 'Jt' methods must be overloaded.
    """

    def __init__(self):
        raise NotImplementedError('PrototypeFunctional.__init__ needs to be \
                overloaded')

    def __add__(self, other):
        ''' method to add functionals together '''
        return CombinedFunctional([self, other])

    def __sub__(self, other):
        ''' method to subtract one functional from another '''
        return CombinedFunctional([self, -other])

    def __mul__(self, other):
        ''' method to scale a functional '''
        return ScaledFunctional(self, other)

    def __rmul__(self, other):
        ''' preserves commutativity of scaling '''
        return ScaledFunctional(self, other)

    def __neg__(self):
        ''' implements the negative of the functional '''
        return -1 * self

    def Jt(self, state, tf):
        r'''This method should return the form which computes the functional's
        contribution for one timelevel.'''
        raise NotImplementedError('PrototypeFunctional.Jt needs to be \
                overloaded.')


class CombinedFunctional(PrototypeFunctional):
    ''' Constructs a single combined functional by adding one functional to
    another.
    '''

    def __init__(self, functional_list):
        for functionals in functional_list:
            assert isinstance(functionals, PrototypeFunctional)
        self.functional_list = functional_list

    def Jt(self, state, tf):
        '''Returns the form which computes the combined functional.'''
        combined_functional = sum([functional.Jt(state, tf) for functional in \
            self.functional_list])
        return combined_functional


class ScaledFunctional(PrototypeFunctional):
    '''Scales the functional
    '''

    def __init__(self, functional, scaling_factor):
        assert isinstance(functional, PrototypeFunctional)
        assert isinstance(scaling_factor, int) or isinstance(scaling_factor, float)
        self.functional = functional
        self.scaling_factor = scaling_factor

    def Jt(self, state, tf):
        '''Returns the form which computes the combined functional.'''
        scaled_functional = self.scaling_factor * self.functional.Jt(state, tf)
        return scaled_functional

