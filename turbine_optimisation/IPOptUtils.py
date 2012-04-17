import numpy
import memoize

# The wrapper class of the objective/constaint functions that as required by the ipopt package
class  IPOptFunction(object):

  def __init__(self):
    pass

  def objective(self, x):
    ''' The objective function evaluated at x. '''
    print "The objective_user function must be overloaded."

  def gradient(self, x):
    ''' The gradient of the objective function evaluated at x. '''
    print "The gradient_user function must be overloaded."

  def constraints(self, x):
    ''' The constraint functions evaluated at x. '''
    return numpy.array([])

  def jacobian(self, x):
    ''' The Jacobian of the constraint functions evaluated at x. '''
    return (numpy.array([]), numpy.array([]))

def position_constraints(params, spacing_sides = 0):
  ''' This function returns the constraints to ensure that the turbine positions remain inside the domain. '''
  n = len(params["turbine_pos"])
  lc = []
  lb_x = params["turbine_x"]/2 
  lb_y = params["turbine_y"]/2 + spacing_sides
  ub_x = params["basin_x"] - params["turbine_x"]/2 
  ub_y = params["basin_y"] - params["turbine_y"]/2 - spacing_sides
  
  # The control variable is ordered as [t1_x, t1_y, t2_x, t2_y, t3_x, ...]
  lb = n * [lb_x, lb_y]
  ub = n * [ub_x, ub_y]
  return lb, ub 

