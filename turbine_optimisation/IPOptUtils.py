import numpy

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

def position_constraints(config, site_x_start = 0, site_x_end = None, site_y_start = 0, site_y_end = None):
    ''' This function returns the constraints to ensure that the turbine positions remain inside the domain plus an optional spacing. '''
    if not site_x_end:
        site_x_end = config.domain.basin_x 
    if not site_y_end:
        site_y_end = config.domain.basin_y 

    n = len(config.params["turbine_pos"])
    lc = []
    lb_x = site_x_start + config.params["turbine_x"]/2 
    lb_y = site_y_start + config.params["turbine_y"]/2 
    ub_x = site_x_end - config.params["turbine_x"]/2 
    ub_y = site_y_end - config.params["turbine_y"]/2 
  
    # The control variable is ordered as [t1_x, t1_y, t2_x, t2_y, t3_x, ...]
    lb = n * [lb_x, lb_y]
    ub = n * [ub_x, ub_y]
    return lb, ub 

def get_minimum_distance_constraint_func(config, min_distance = None):
    if config.params['controls'] != ['turbine_pos']:
        raise NotImplementedError, "Inequality contraints are currently only supported if turbine_pos are the only controls"

    if not min_distance:
        min_distance = 2*max(config.params["turbine_x"], config.params["turbine_y"])

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
