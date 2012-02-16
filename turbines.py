import numpy
from dolfin import *
from math import log

class RectangleTurbines(Expression):
    '''The turbines are modeled by rectangles of size turbine_length*scalefac and turbine_width*scalefac.
       Scalefac is default to 1.'''
    def __init__(self, config, scalefac=1.0, *args, **kwargs):
      self.config = config
      self.scalefac = scalefac
      super(RectangleTurbines, self).__init__(args, kwargs)

    def eval(self, values, x):
        params = self.config.params
        scalefac = self.scalefac
        friction = 0.0
        if len(params["turbine_pos"]) >0:
          # Check if x lies in a position where a turbine is deployed and if, then increase the friction
          x_pos = numpy.array(params["turbine_pos"])[:,0] 
          x_pos_low = x_pos-params["turbine_length"]*scalefac/2
          x_pos_high = x_pos+params["turbine_length"]*scalefac/2

          y_pos = numpy.array(params["turbine_pos"])[:,1] 
          y_pos_low = y_pos-params["turbine_width"]*scalefac/2
          y_pos_high = y_pos+params["turbine_width"]*scalefac/2

          # active_turbines is a boolean array that whose i'th element is true if the ith turbine is present at point x
          active_turbines = (x_pos_low < x[0]) & (x_pos_high > x[0]) & (y_pos_low < x[1]) & (y_pos_high > x[1])
          active_turbines_indices = numpy.where(active_turbines == True)[0]

          for i in active_turbines_indices:
            friction += params["turbine_friction"][i] 

        values[0] = friction 


class GaussianTurbines(Expression):
    '''The turbines are modeled by a gaussian curve size turbine_length*scalefac and turbine_width*scalefac.
       Scalefac is default to 1.'''
    def __init__(self, config, scalefac=1.0, *args, **kwargs):
      self.config = config
      self.scalefac = scalefac
      super(GaussianTurbines, self).__init__(args, kwargs)

    def eval(self, values, x):
        params = self.config.params
        scalefac = self.scalefac
        friction = 0.0
        cut_off = 0.05 # Specifies which value the gaussian curve is supposed to have at the edges of the turbine size (as percentage of the maximal value)
                       # This parameter influences the steepnes of the gaussian function.
        if len(params["turbine_pos"]) >0:
          # Check if x lies in a position where a turbine is deployed and if, then increase the friction
          x_pos = numpy.array(params["turbine_pos"])[:,0] 
          x_pos_low = x_pos-params["turbine_length"]/scalefac/2
          x_pos_high = x_pos+params["turbine_length"]/scalefac/2

          y_pos = numpy.array(params["turbine_pos"])[:,1] 
          y_pos_low = y_pos-params["turbine_width"]/scalefac/2
          y_pos_high = y_pos+params["turbine_width"]/scalefac/2

          # active_turbines is a boolean array that whose i'th element is true if the ith turbine is present at point x
          active_turbines = (x_pos_low < x[0]) & (x_pos_high > x[0]) & (y_pos_low < x[1]) & (y_pos_high > x[1])
          active_turbines_indices = numpy.where(active_turbines == True)[0]

          for i in active_turbines_indices:
            gaussian = exp(-0.5 * (x[0]-x_pos[i])**2 * (-2*log(0.05)) / ((0.5*params["turbine_length"])**2) - 0.5 * (x[1]-y_pos[i])**2 * (-2*log(0.05)) / ((0.5*params["turbine_width"])**2))
            friction += gaussian*params["turbine_friction"][i]

        values[0] = friction
