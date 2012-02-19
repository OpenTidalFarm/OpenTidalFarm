import numpy
import sw_lib
from dolfin import *
from math import log

class Turbines(Expression):
    def __init__(self, params, *args, **kwargs):
      self.params = sw_lib.parameters(params)
      super(Turbines, self).__init__(args, kwargs)

    # The turbine functions will be evaluated between (-1..1) x (-1..1) and should return function values from [0..1].
    def constant_turbine_function(self, x):
      '''The turbines are modeled by rectangles with constant friction. '''
      return 1.0

    def gaussian_turbine_function(self, x):
      '''The turbines are modeled by a gaussian curve. ''' 
      return exp(-0.5 * (x[0]**2 + x[1]**2) * (-2*log(0.05)) )

    def bump_turbine_function(self, x):
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

    def turbine_function(self, params):
      ''' Returns the turbine function using the parameters. '''
      functions = {'GaussianTurbine': self.gaussian_turbine_function, 'ConstantTurbine': self.constant_turbine_function, 'BumpTurbine': self.bump_turbine_function }
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

          f = self.turbine_function(params)
          for i in active_turbines_indices:
            x_unit = (x[0]-x_pos[i]) / (0.5*params["turbine_length"])
            y_unit = (x[1]-y_pos[i]) / (0.5*params["turbine_width"])
            friction += f([x_unit, y_unit])*params["turbine_friction"][i] 
        values[0] = friction 
