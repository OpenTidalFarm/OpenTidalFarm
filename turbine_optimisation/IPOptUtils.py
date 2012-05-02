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

def position_constraints(params, spacing_sides = 0, spacing_left = 0, spacing_right = 0):
  ''' This function returns the constraints to ensure that the turbine positions remain inside the domain. '''
  n = len(params["turbine_pos"])
  lc = []
  lb_x = params["turbine_x"]/2 + spacing_left 
  lb_y = params["turbine_y"]/2 + spacing_sides
  ub_x = params["basin_x"] - params["turbine_x"]/2 + spacing_right
  ub_y = params["basin_y"] - params["turbine_y"]/2 - spacing_sides
  
  # The control variable is ordered as [t1_x, t1_y, t2_x, t2_y, t3_x, ...]
  lb = n * [lb_x, lb_y]
  ub = n * [ub_x, ub_y]
  return lb, ub 

def get_minimum_distance_constraint_func(config, min_distance = 40):
    if config.params['controls'] != ['turbine_pos']:
        raise NotImplementedError, "Inequality contraints are currently only supported if turbine_pos are the only controls"

    def l2norm(x):
        return sum([v**2 for v in x])

    def f_ieqcons(m):
        ieqcons = []
        for i in range(len(m)/2):                                                                           
            for j in range(len(m)/2):                                                                       
                if i <= j:
                    continue
                ieqcons.append(l2norm( [m[2*i]-m[2*j], m[2*i+1]-m[2*j+1]] ) - min_distance**2)              
        return numpy.array(ieqcons)

    def fprime_ieqcons(m):
        ieqcons = []
        for i in range(len(m)/2):
            for j in range(len(m)/2):
                if i <= j:
                    continue
                prime_ieqcons = numpy.zeros(len(m))

                prime_ieqcons[2*i] = 2*(m[2*i]-m[2*j])
                prime_ieqcons[2*j] = -2*(m[2*i]-m[2*j])
                prime_ieqcons[2*i+1] = 2*(m[2*i+1]-m[2*j+1])
                prime_ieqcons[2*j+1] = -2*(m[2*i+1]-m[2*j+1])

                ieqcons.append(prime_ieqcons)
        return numpy.array(ieqcons)

    return f_ieqcons, fprime_ieqcons 
