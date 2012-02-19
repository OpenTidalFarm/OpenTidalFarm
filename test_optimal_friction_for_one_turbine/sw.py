import sys
import sw_config 
import sw_lib
import numpy
import Memoize
import IPOptUtils
from functionals import DefaultFunctionalWithoutControlDependency 
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array
from turbines import *

# Global counter variable for vtk output
count = 0

def default_config():
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
  config.params["turbine_friction"] = 12.0*numpy.ones(len(config.params["turbine_pos"]))

  config.params["turbine_length"] = 800
  config.params["turbine_width"] = 800

  # Now create the turbine measure
  config.initialise_turbines_measure()
  return config

def initial_control(config):
  numpy.random.seed(41) 
  return numpy.random.rand(len(config.params['turbine_friction']))

def j_and_dj(x):
  adjointer.reset()
  adj_variables.__init__()
  
  set_log_level(30)
  debugging["record_all"] = True

  W=sw_lib.p1dgp2(config.mesh)

  # Get initial conditions
  state=Function(W)
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U)
  # Apply the control

  # Set up the turbine friction field using the provided control variable
  turbine_friction_orig = config.params["turbine_friction"]
  config.params["turbine_friction"] = x * turbine_friction_orig
  tf.interpolate(config.params["turbine_model"](config.params))
  config.params["turbine_friction"] = turbine_friction_orig 

  global count
  count+=1
  sw_lib.save_to_file_scalar(tf, "turbines_"+str(count))

  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)
  functional = DefaultFunctionalWithoutControlDependency(config.params)

  # Solve the shallow water system
  j, djdm, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)
  #print "Layout power extraction: ", j/1000000, " MW."
  #print "Which is equivalent to a average power generation of: ",  j/1000000/(config.params["current_time"]-config.params["start_time"]), " MW"

  #sw_lib.replay(state, config.params)

  J = TimeFunctional(functional.Jt(state))
  adj_state = sw_lib.adjoint(state, config.params, J, until=1)

  #sw_lib.save_to_file(adj_state, "adjoint")

  # we have dJ/dx = (\partial J)/(\partial turbine_friction) * (d turbine_friction) / d x +
  #                 + \partial J / \partial x
  #               = adj_state * turbine_friction
  #                 + \partial J / \partial x
  # In this particular case, j = \sum_t(functional) and \partial functional / \partial x = funtional/x. Hence we haev \partial J / \partial x = j/x
  dj = numpy.zeros(len(config.params["turbine_friction"]))
  v = adj_state.vector()
  for n in range(len(dj)):
    turbine_friction_orig = config.params["turbine_friction"]
    x = numpy.zeros(len(dj))
    x[n] = 1.0
    config.params["turbine_friction"] = x*config.params["turbine_friction"]
    tf.interpolate(config.params["turbine_model"](config.params))
    dj[n] = v.inner(tf.vector()) 
    config.params["turbine_friction"] = turbine_friction_orig 
  return j, dj 

j_and_dj_mem = Memoize.MemoizeMutable(j_and_dj)
def j(x):
  j = j_and_dj_mem(x)[0]*10**-13
  print 'Evaluating j(', x.__repr__(), ')=', j
  return j 

def dj(x):
  dj = j_and_dj_mem(x)[1]*10**-13
  print 'Evaluating dj(', x.__repr__(), ')=', dj
  return dj

config = default_config()
x0 = initial_control(config)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
p = numpy.random.rand(len(config.params['turbine_friction']))
minconv = test_gradient_array(j, dj, x0, seed=0.0001, perturbation_direction=p)
if minconv < 1.98:
  print "The gradient taylor remainder test failed."
  sys.exit(1)

opt_package = 'ipopt'

if opt_package == 'ipopt':
  # If this option does not produce any ipopt outputs, delete the ipopt.opt file
  import ipopt 
  g = lambda x: []
  dg = lambda x: []

  f = IPOptUtils.IPOptFunction()
  # Overwrite the functional and gradient function with our implementation
  f.objective= j 
  f.gradient= dj 

  nlp = ipopt.problem(len(x0), 
                      0, 
                      f, 
                      numpy.zeros(len(x0)), 
                      100*numpy.ones(len(x0)))
  nlp.addOption('mu_strategy', 'adaptive')
  nlp.addOption('tol', 1e-7)
  nlp.addOption('print_level', 5)
  nlp.addOption('check_derivatives_for_naninf', 'yes')
  # A -1.0 scaling factor transforms the min problem to a max problem.
  nlp.addOption('obj_scaling_factor', -1.0)
  # Use an approximate Hessian since we do not have second order information.
  nlp.addOption('hessian_approximation', 'limited-memory')

  x, info = nlp.solve(x0)
  print info['status_msg']
  print "Solution of the primal variables: x=%s\n" % repr(x) 
  print "Solution of the dual variables: lambda=%s\n" % repr(info['mult_g'])
  print "Objective=%s\n" % repr(info['obj_val'])

  if info['status'] != 0 or max(x) > 0: 
    print "The optimisation algorithm did not find the correct solution."
    sys.exit(1) 
  else:
    exit_code = 0

if opt_package == 'pyipopt':
  import pyipopt
  g = lambda x: []
  dg = lambda x: []
  nlp = pyipopt.create(len(x0), 
                       numpy.zeros(len(config.params["turbine_friction"])), 
                       100*numpy.ones(len(config.params["turbine_friction"])), 
                       0, 
                       numpy.array([]), 
                       numpy.array([]), 
                       0, 0, j, dj, g, dg)

  x, zl, zu, obj, status = nlp.solve(x0)
  nlp.close()
  print "Solution of the primal variables, x"
  print "x", x

  print "Objective value"
  print "f(x*) =", obj


if opt_package == 'openopt':
  from openopt import NLP

  # Restrict the control to positive friction values
  c = [lambda x: -x]
  dc = [lambda x: -numpy.ones(len(x))]

  p = NLP(j, x0, df=dj,# c=c, dc=dc,# h=h,  dh=dh,  A=A,  b=b,  Aeq=Aeq,  beq=beq, 
      #lb=lb, ub=ub, gtol=gtol, contol=contol, iprint = 50, maxIter = 10000, 
      maxFunEvals = 1e7, name = 'NLP_1')
  p.plot = True
  p.goal ='max'
  #p.checkdf()
  #p.checkdc()

  # For solvers see: http://openopt.org/NLP
  solver = 'ipopt'
  #solver = 'scipy_slsqp'
  #solver = 'ralg'
  r = p.solve(solver, plot=0)
