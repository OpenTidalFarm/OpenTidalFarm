"""
.. module:: coupler
   :synopsis: Couples the thetis solver to the opentidalfarm

"""
from helpers import FrozenClass

# Absolute import for type checking only
from opentidalfarm import Farm

class CouplerParameters(FrozenClass):
    pass

class Coupler(object):

    def __init__(self, thetis_solver, farm, coupler_parameters):
        
        # TODO type checking for thetis - needs to be a thetis solver and
        # 2 dimensional shallow water etc...
#        if not isinstance(thetis_solver, XXX):
#            raise TypeError('thetis_solver must be of type XXX')

        if not isinstance(farm, Farm):
            raise TypeError('farm must be of type Farm')

        if not isinstance(coupler_parameters, CouplerParameters):
            raise TypeError('coupler_parameters must be of type \
                            CouplerParameters')

        self.solver = thetis_solver
        self.farm = farm
        self.params = coupler_parameters

        print 'Inside coupler'
        from IPython import embed; embed()

    def solve(self):
        """ Solves the coupled flow / farm problem
        """
#        self.solver.options.quadratic_drag = self.farm.friction_function
        self.solver.iterate()



    @staticmethod
    def default_parameters():
        """ Return the default parameters for the :class: 'Coupler'.
        """
        return CouplerParameters()

