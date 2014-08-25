from math import exp
from dolfin import *


def SinusoidalInitialCondition(eta0, k, depth, start_time):
    """Returns an expression that can be used as initial condition for a channel
    with a sinusoidal forcing"""
    

    class SinusoidalExpr(Expression):
        '''This class implements the Expression class for the shallow water initial condition.'''
        def __init__(self):
            pass

        def eval(self, values, X):
            g = 9.81

            values[0] = eta0 * sqrt(g / depth) * cos(k * X[0] - sqrt(g * depth) * k * start_time)
            values[1] = 0.
            values[2] = eta0 * cos(k * X[0] - sqrt(g * depth) * k * start_time)

        def value_shape(self):
            return (3,)
    return SinusoidalExpr()


def BumpInitialCondition(config):

    class BumpExpr(Expression):
        '''This class implements a initial condition with a bump velocity profile.
           With that we know that the optimal turbine location must be in the center of the domain. '''

        def bump_function(self, x):
            '''The velocity is initially a bump function (a smooth function with limited support):
                       /  e**-1/(1-x**2)   for |x| < 1
              psi(x) = |
                       \  0   otherwise
              For more information see http://en.wikipedia.org/wiki/Bump_function
            '''
            if x[0] ** 2 < 1 and x[1] ** 2 < 1:
                bump = exp(-1.0 / (1.0 - x[0] ** 2))
                bump *= exp(-1.0 / (1.0 - x[1] ** 2))
                bump /= exp(-1) ** 2
            else:
                bump = 0.0
            return bump

        def eval(self, values, X):
            x_unit = 2 * (config.domain.basin_x - X[0]) / config.domain.basin_x - 1.0
            y_unit = 2 * (config.domain.basin_y - X[1]) / config.domain.basin_y - 1.0

            values[0] = self.bump_function([x_unit, y_unit])
            values[1] = 0
            values[2] = 0

        def value_shape(self):
            return (3,)

    return BumpExpr()


def ConstantFlowInitialCondition(config, val=[1e-19, 0, 0, 0]):
    class ConstantFlow(Expression):
        def eval(self, values, X):
            values[0] = val[0]
            values[1] = val[1]
            values[2] = val[2]

        def value_shape(self):
            return (3,)

    return ConstantFlow()
