import numpy
import Memoize

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

def position_constraints(params):
  ''' This function returns the constraints for valid turbine positions. '''
  l = len(params["turbine_pos"])
  lc = []
  lb_x = params["turbine_x"] 
  #lb_y = 
  for i in range(l):
    lc.append([lb, ub])
  
