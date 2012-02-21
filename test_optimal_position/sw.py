import sys
import cProfile
import pstats
import sw_config 
import sw_lib
import numpy
import Memoize
import IPOptUtils
from functionals import DefaultFunctional
from sw_utils import test_initial_condition_adjoint, test_gradient_array
from turbines import *
from mini_model import *
from dolfin import *
from dolfin_adjoint import *

# Global counter variable for vtk output
count = 0

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=20, ny=10)
  config.params["verbose"] = 0

  # dt is used in the functional only, so we set it here to 1.0
  config.params["dt"] = 1.0
  # Turbine settings
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [[500., 200.]]
  config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 800
  config.params["turbine_y"] = 800

  return config

def initial_control(config):
  # We use the current turbine settings as the intial control
  res = numpy.reshape(config.params['turbine_pos'], -1).tolist()
  return numpy.array(res)

class BumpInitialConditions(Expression):
  '''This class implements a initial condition with a bump velocity profile.'''
  def __init__(self, params):
    self.params = params

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

  def eval(self, values, X):
    params = self.params
    x_unit = 2*(params["basin_x"]-X[0])/params["basin_x"]-1.0
    y_unit = 2*(params["basin_y"]-X[1])/params["basin_y"]-1.0

    values[0] = self.bump_function([x_unit, y_unit]) 
    values[1] = 0.
    values[2] = 0.0 
  def value_shape(self):
    return (3,)

def j_and_dj(m):
  adjointer.reset()
  adj_variables.__init__()

  # Change the control variables to the config parameters
  config.params["turbine_pos"] = numpy.reshape(m, (-1, 2))

  set_log_level(30)
  debugging["record_all"] = True

  W=sw_lib.p1dgp2(config.mesh)
  state=Function(W)
  state.interpolate(BumpInitialConditions(config.params))

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U) # The turbine function
  tfd = Function(U) # The derivative turbine function

  # Set up the turbine friction field using the provided control variable
  tf.interpolate(Turbines(config.params))

  global count
  count+=1
  sw_lib.save_to_file_scalar(tf, "turbines_t=."+str(count)+".x")

  A, M = construct_mini_model(W, config.params, tf)

  functional = DefaultFunctional(config.params)

  # Solve the shallow water system
  j, djdm, state = mini_model(A, M, state, config.params, functional)
  J = TimeFunctional(functional.Jt(state))
  adj_state = sw_lib.adjoint(state, config.params, J, until=1) # The first annotation is the idendity operator for the turbine field

  # Let J be the functional, m the parameter and u the solution of the PDE equation F(u) = 0.
  # Then we have 
  # dJ/dm = (\partial J)/(\partial u) * (d u) / d m + \partial J / \partial m
  #               = adj_state * \partial F / \partial u + \partial J / \partial m
  # In this particular case m = turbine_friction, J = \sum_t(ft) 
  dj = [] 
  v = adj_state.vector()
  # Compute the derivatives with respect to the turbine friction
  for n in range(len(config.params["turbine_friction"])):
    tfd.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector='turbine_friction'))
    dj.append( v.inner(tfd.vector()) )

  # Compute the derivatives with respect to the turbine position
  for n in range(len(config.params["turbine_pos"])):
    for var in ('turbine_pos_x', 'turbine_pos_y'):
      tfd.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector=var))
      dj.append( v.inner(tfd.vector()) )
  dj = numpy.array(dj)  
  
  # Now add the \partial J / \partial m term
  dj += djdm

  return j, dj 

j_and_dj_mem = Memoize.MemoizeMutable(j_and_dj)
def j(m):
  j = j_and_dj_mem(m)[0]
  print 'Evaluating j(', m.__repr__(), ')=', j
  return j 

def dj(m):
  dj = j_and_dj_mem(m)[1]
  print 'Evaluating dj(', m.__repr__(), ')=', dj
  # Return only the derivatives with respect to the position
  return dj[len(config.params['turbine_friction']):]

config = default_config()
m0 = initial_control(config)

p = numpy.random.rand(len(m0))
minconv = test_gradient_array(j, dj, m0, seed=0.001, perturbation_direction=p)
if minconv < 1.99:
  print "The gradient taylor remainder test failed."
  sys.exit(1)

opt_package = 'ipopt'

if opt_package == 'ipopt':
  # If this option does not produce any ipopt outputs, delete the ipopt.opt file
  import ipopt 
  g = lambda m: []
  dg = lambda m: []

  f = IPOptUtils.IPOptFunction()
  # Overwrite the functional and gradient function with our implementation
  f.objective= j 
  f.gradient= dj 

  nlp = ipopt.problem(len(m0), 
                      0, 
                      f, 
                      numpy.zeros(len(m0)), 
                      3000.*numpy.ones(len(m0)))
  nlp.addOption('mu_strategy', 'adaptive')
  nlp.addOption('tol', 1e-9)
  nlp.addOption('print_level', 5)
  nlp.addOption('check_derivatives_for_naninf', 'yes')
  # A -1.0 scaling factor transforms the min problem to a max problem.
  nlp.addOption('obj_scaling_factor', -1.0)
  # Use an approximate Hessian since we do not have second order information.
  nlp.addOption('hessian_approximation', 'limited-memory')
  nlp.addOption('max_iter', 25)

  m, info = nlp.solve(m0)
  print info['status_msg']
  print "Solution of the primal variables: m=%s\n" % repr(m) 
  print "Solution of the dual variables: lambda=%s\n" % repr(info['mult_g'])
  print "Objective=%s\n" % repr(info['obj_val'])

  exit_code = 1
  if info['status'] != 0: 
    print "The optimisation algorithm did not find a solution."
  elif abs(m[0]-1500) > 40:
    print "The optimisation algorithm did not find the optimal x position:", m[0] , "instead of 1500."
  elif abs(m[1]-500) > 0.4:
    print "The optimisation algorithm did not find the optimal y position:", m[1] , "instead of 500."
  else:
    exit_code = 0

sys.exit(exit_code) 
