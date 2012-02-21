import sys
import sw_config 
import sw_lib
import numpy
import Memoize
import IPOptUtils
from functionals import DefaultFunctional
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array
from turbines import *

# Global counter variable for vtk output
count = 0

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=20, ny=10)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  config.params["finish_time"] = 2./4*period
  config.params["dt"] = config.params["finish_time"]/10
  print "Wave period (in h): ", period/60/60 
  config.params["dump_period"] = 1000
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4 

  # Turbine settings
  config.params["friction"] = 0.0025
  config.params["turbine_pos"] = [[500., 500.], [1500., 500.], [2500., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 800
  config.params["turbine_y"] = 800

  return config

def initial_control(config):
  # We use the current turbine settings as the intial control
  res = config.params['turbine_friction'].tolist()
  #res += numpy.reshape(config.params['turbine_pos'], -1).tolist()
  return numpy.array(res)

def j_and_dj(m):
  adjointer.reset()
  adj_variables.__init__()

  # Change the control variables to the config parameters
  config.params["turbine_friction"] = m[:len(config.params["turbine_friction"])]
  mp = m[len(config.params["turbine_friction"]):]
  #config.params["turbine_pos"] = numpy.reshape(mp, (-1, 2))

  set_log_level(30)
  debugging["record_all"] = True

  W=sw_lib.p1dgp2(config.mesh)

  # Get initial conditions
  state=Function(W)
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U) # The turbine function
  tfd = Function(U) # The derivative turbine function

  # Set up the turbine friction field using the provided control variable
  tf.interpolate(Turbines(config.params))

  global count
  count+=1
  sw_lib.save_to_file_scalar(tf, "turbines_"+str(count))

  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)

  functional = DefaultFunctional(config.params)

  # Solve the shallow water system
  j, djdm, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)
  #print "Layout power extraction: ", j/1000000, " MW."
  #print "Which is equivalent to a average power generation of: ",  j/1000000/(config.params["current_time"]-config.params["start_time"]), " MW"

  #sw_lib.replay(state, config.params)

  J = TimeFunctional(functional.Jt(state))
  adj_state = sw_lib.adjoint(state, config.params, J, until=1)

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
  j = j_and_dj_mem(m)[0]*10**-13
  print 'Evaluating j(', m.__repr__(), ')=', j
  return j 

def dj(m):
  dj = j_and_dj_mem(m)[1]*10**-13
  print 'Evaluating dj(', m.__repr__(), ')=', dj
  # Return only the derivatives with respect to the friction
  return dj[:len(config.params['turbine_friction'])]

config = default_config()
m0 = initial_control(config)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
p = numpy.random.rand(len(config.params['turbine_friction']))
minconv = test_gradient_array(j, dj, m0, seed=0.001, perturbation_direction=p)
if minconv < 1.98:
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
                      100*numpy.ones(len(m0)))
  nlp.addOption('mu_strategy', 'adaptive')
  nlp.addOption('tol', 1e-7)
  nlp.addOption('print_level', 5)
  nlp.addOption('check_derivatives_for_naninf', 'yes')
  # A -1.0 scaling factor transforms the min problem to a max problem.
  nlp.addOption('obj_scaling_factor', -1.0)
  # Use an approximate Hessian since we do not have second order information.
  nlp.addOption('hessian_approximation', 'limited-memory')

  m, info = nlp.solve(m0)
  print info['status_msg']
  print "Solution of the primal variables: m=%s\n" % repr(m) 
  print "Solution of the dual variables: lambda=%s\n" % repr(info['mult_g'])
  print "Objective=%s\n" % repr(info['obj_val'])

  if info['status'] != 0 or max(m) > 0: 
    print "The optimisation algorithm did not find the correct solution."
    sys.exit(1) 
  else:
    exit_code = 0

#if opt_package == 'pyipopt':
#  import pyipopt
#  g = lambda m: []
#  dg = lambda m: []
#  nlp = pyipopt.create(len(m0), 
#                       numpy.zeros(len(config.params["turbine_friction"])), 
#                       100*numpy.ones(len(config.params["turbine_friction"])), 
#                       0, 
#                       numpy.array([]), 
#                       numpy.array([]), 
#                       0, 0, j, dj, g, dg)
#
#  m, zl, zu, obj, status = nlp.solve(m0)
#  nlp.close()
#  print "Solution of the primal variables, m"
#  print "m", m
#
#  print "Objective value"
#  print "f(m*) =", obj
#
#if opt_package == 'openopt':
#  from openopt import NLP
#
#  # Restrict the control to positive friction values
#  c = [lambda m: -m]
#  dc = [lambda m: -numpy.ones(len(m))]
#
#  p = NLP(j, m0, df=dj,# c=c, dc=dc,# h=h,  dh=dh,  A=A,  b=b,  Aeq=Aeq,  beq=beq, 
#      #lb=lb, ub=ub, gtol=gtol, contol=contol, iprint = 50, maxIter = 10000, 
#      maxFunEvals = 1e7, name = 'NLP_1')
#  p.plot = True
#  p.goal ='max'
#  #p.checkdf()
#  #p.checkdc()
#
#  # For solvers see: http://openopt.org/NLP
#  solver = 'ipopt'
#  #solver = 'scipy_slsqp'
#  #solver = 'ralg'
#  r = p.solve(solver, plot=0)
