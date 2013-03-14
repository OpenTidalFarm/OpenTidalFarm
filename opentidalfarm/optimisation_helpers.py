import numpy
from dolfin import info_blue, Constant
from helpers import info, info_green, info_red, info_blue

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

def deploy_turbines(config, nx, ny, friction=21.):
    ''' Generates an array of initial turbine positions with nx x ny turbines homonginuosly distributed over the site with the specified dimensions. '''
    turbine_pos = []
    for x_r in numpy.linspace(config.domain.site_x_start + 0.5*config.params["turbine_x"], config.domain.site_x_end - 0.5*config.params["turbine_y"], nx):
        for y_r in numpy.linspace(config.domain.site_y_start + 0.5*config.params["turbine_x"], config.domain.site_y_end - 0.5*config.params["turbine_y"], ny):
            turbine_pos.append((float(x_r), float(y_r)))
    config.set_turbine_pos(turbine_pos, friction)
    info_blue("Deployed " + str(len(turbine_pos)) + " turbines.")
    return turbine_pos

def position_constraints(config):
    ''' This function returns the constraints to ensure that the turbine positions remain inside the domain plus an optional spacing. '''

    n = len(config.params["turbine_pos"])
    lc = []
    lb_x = config.domain.site_x_start + config.params["turbine_x"]/2 
    lb_y = config.domain.site_y_start + config.params["turbine_y"]/2 
    ub_x = config.domain.site_x_end - config.params["turbine_x"]/2 
    ub_y = config.domain.site_y_end - config.params["turbine_y"]/2 

    if not lb_x < ub_x or not lb_y < ub_y:
        raise ValueError, "Lower bound is larger than upper bound. Is your domain large enough?"
  
    # The control variable is ordered as [t1_x, t1_y, t2_x, t2_y, t3_x, ...]
    lb = n * [Constant(lb_x), Constant(lb_y)]
    ub = n * [Constant(ub_x), Constant(ub_y)]
    return lb, ub 

def friction_constraints(config, lb = 0.0, ub = None):
    ''' This function returns the constraints to ensure that the turbine friction controls remain reasonable. '''

    if ub != None and not lb < ub:
        raise ValueError, "Lower bound is larger than upper bound."
    
    if ub == None:
      ub = 10**12

    n = len(config.params["turbine_pos"])
    return n * [Constant(lb)], n * [Constant(ub)] 

def get_minimum_distance_constraint_func(config, min_distance = None):
    if not 'turbine_pos' in config.params['controls']:
        raise NotImplementedError, "Inequality contraints for the distance only make sense if the turbine positions are control variables."

    if not min_distance:
        min_distance = 1.5*max(config.params["turbine_x"], config.params["turbine_y"])

    def l2norm(x):
        return sum([v**2 for v in x])

    def f_ieqcons(m):
        ieqcons = []
        if len(config.params['controls']) == 2:
        # If the controls consists of the the friction and the positions, then we need to first extract the position part
          assert(len(m)%3 == 0)
          m = m[len(m)/3:]

        for i in range(len(m)/2):                                                                           
            for j in range(len(m)/2):                                                                       
                if i <= j:
                    continue
                ieqcons.append(l2norm( [m[2*i]-m[2*j], m[2*i+1]-m[2*j+1]] ) - min_distance**2)              
        return numpy.array(ieqcons)

    def fprime_ieqcons(m):
        ieqcons = []
        if len(config.params['controls']) == 2:
          # If the controls consists of the the friction and the positions, then we need to first extract the position part
          assert(len(m)%3 == 0)
          m = m[len(m)/3:]
          mf_len = len(m)/2
        else:
          mf_len = 0

        for i in range(len(m)/2):
            for j in range(len(m)/2):
                if i <= j:
                    continue
                prime_ieqcons = numpy.zeros(mf_len + len(m))

                # The control vector contains the friction coefficients first, so we need to shift here
                prime_ieqcons[mf_len + 2*i] = 2*(m[2*i]-m[2*j])
                prime_ieqcons[mf_len + 2*j] = -2*(m[2*i]-m[2*j])
                prime_ieqcons[mf_len + 2*i+1] = 2*(m[2*i+1]-m[2*j+1])
                prime_ieqcons[mf_len + 2*j+1] = -2*(m[2*i+1]-m[2*j+1])

                ieqcons.append(prime_ieqcons)
        return numpy.array(ieqcons)

    return {'type': 'ineq', 'fun': f_ieqcons, 'jac': fprime_ieqcons} 
