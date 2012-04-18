from math import exp
from dolfin import *

def SinusoidalInitialCondition(config):
    params = config.params

    class SinusoidalExpr(Expression):
        '''This class implements the Expression class for the shallow water initial condition.'''
        def __init__(self):
            pass
        def eval(self, values, X):
            eta0 = params['eta0']
            g = params['g']
            k = params['k']
            depth = params['depth']
            start_time = params["start_time"]

            values[0] = 1./depth * eta0 * sqrt(g * depth) * cos(k * X[0] - sqrt(g * depth) * k * start_time)
            values[1] = 0.
            values[2] = eta0 * cos(k * X[0] - sqrt(g * depth) * k * start_time)
        def value_shape(self):
            return (3,)
    return SinusoidalExpr

def BumpInitialCondition(config):
    params = config.params

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
        bump = exp(-1.0/(1.0-x[0]**2)) 
        bump *= exp(-1.0/(1.0-x[1]**2)) 
        bump /= exp(-1)**2
        return bump

      def eval(self, values, X):
        x_unit = 2*(params["basin_x"]-X[0])/params["basin_x"]-1.0
        y_unit = 2*(params["basin_y"]-X[1])/params["basin_y"]-1.0

        values[0] = self.bump_function([x_unit, y_unit]) 
        values[1] = 0.
        values[2] = 0.0 
      def value_shape(self):
        return (3,)

    return BumpExpr

def ConstantFlowInitialCondition(config):
    class ConstantFlow(Expression):
        def __init__(self):
            self.start_time = config.params["start_time"]
            self.depth = config.params["depth"]
            self.k = config.params["k"]
            self.g = config.params["g"]
            self.eta0 = config.params["eta0"]

        def eval(self, values, X):
            values[0] = 1e-19
            values[1] = 0.
            values[2] = 0. 
        def value_shape(self):
            return (3,)

    return ConstantFlow 
