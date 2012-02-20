import numpy
import sw_lib
from dolfin import *
from math import log

class Turbines(Expression):
    def __init__(self, params, derivative_index_selector=-1,  derivative_var_selector=None, *args, **kwargs):
      ''' If the derivative selector is i >= 0, the Expression will compute the derivative of the turbine with index i with respect 
          to either the x or y coorinate or its friction parameter. '''
      self.params = sw_lib.parameters(params)
      self.derivative_index_selector = derivative_index_selector
      self.derivative_var_selector = derivative_var_selector
      super(Turbines, self).__init__(args, kwargs)

    # The turbine functions will be evaluated between (-1..1) x (-1..1) and should return function values from [0..1].
    def constant_function(self, x):
      '''The turbines are modeled by rectangles with constant friction. '''
      return 1.0

    def gaussian_function(self, x):
      '''The turbines are modeled by a gaussian curve. ''' 
      return exp(-0.5 * (x[0]**2 + x[1]**2) * (-2*log(0.05)) )

    def bump_function(self, x):
      '''The turbines are modeled by the bump function (a smooth function with limited support):
                 /  e**-1/(1-x**2)   for |x| < 1
        psi(x) = |  
                 \  0   otherwise
        For more information see http://en.wikipedia.org/wiki/Bump_function
      '''
      bump = exp(-1.0/(1.0-x[0]**2)) 
      bump *= exp(-1.0/(1.0-x[1]**2)) 
      bump /= exp(-1)**2
      return bump

    def bump_derivative(self, x, d):
      ''' This function computes the derivative of the turbine function with respect to the d'th coordinate. '''
      bump = self.bump_function(x)
      bump *= - 2*x[d] / ((1.0-x[d]**2)**2)
      return bump

    def turbine_function(self, params):
      functions = {'GaussianTurbine': self.gaussian_function, 'ConstantTurbine': self.constant_function, 'BumpTurbine': self.bump_function }
      return functions[params['turbine_model']]

    def turbine_derivative(self, params):
      functions = {'BumpTurbine': self.bump_derivative}
      return functions[params['turbine_model']]

    def eval(self, values, x):
        params = self.params
        friction = 0.0
        if len(params["turbine_pos"]) >0:
          # Check if x lies in a position where a turbine is deployed and if, then increase the friction
          x_pos = numpy.array(params["turbine_pos"])[:,0] 
          x_pos_low = x_pos-params["turbine_length"]/2
          x_pos_high = x_pos+params["turbine_length"]/2

          y_pos = numpy.array(params["turbine_pos"])[:,1] 
          y_pos_low = y_pos-params["turbine_width"]/2
          y_pos_high = y_pos+params["turbine_width"]/2

          # active_turbines is a boolean array that whose i'th element is true if the ith turbine is present at point x
          active_turbines = (x_pos_low < x[0]) & (x_pos_high > x[0]) & (y_pos_low < x[1]) & (y_pos_high > x[1])
          active_turbines_indices = numpy.where(active_turbines == True)[0]

          for i in active_turbines_indices:
            if self.derivative_index_selector < 0:
              # Just compute the evaluation
              f = self.turbine_function(params)

              x_unit = (x[0]-x_pos[i]) / (0.5*params["turbine_length"])
              y_unit = (x[1]-y_pos[i]) / (0.5*params["turbine_width"])
              friction += f([x_unit, y_unit])*params["turbine_friction"][i] 

            elif i == self.derivative_index_selector:
              # Compute the derivative with respect to the specified variable
              i = self.derivative_index_selector
              x_unit = (x[0]-x_pos[i]) / (0.5*params["turbine_length"])
              y_unit = (x[1]-y_pos[i]) / (0.5*params["turbine_width"])

              # Now check with which variable we want to take the derivative with respect to.
              var = self.derivative_var_selector
              if var== 'turbine_friction':
                f = self.turbine_function(params)
                friction += f([x_unit, y_unit])
              elif var in ('turbine_pos_x', 'turbine_pos_y'):
                d = {'turbine_pos_x': 0, 'turbine_pos_y': 1}[var]
                f = self.turbine_derivative(params)
                friction += f([x_unit, y_unit], d)*params["turbine_friction"][i]*(-1.0/(0.5*params["turbine_length"])) # The last multiplier is the derivative of x_unit due to the chain rule
              else: 
                raise ValueError, "Invalid argument for the derivarive variable selector."

        values[0] = friction 
