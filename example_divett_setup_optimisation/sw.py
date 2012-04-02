''' This example optimises the position of three turbines using the hallow water model. '''

import sys
import sw_config 
import sw_lib
import numpy
import Memoize
import IPOptUtils
import ipopt
from animated_plot import *
from functionals import DefaultFunctional, build_turbine_cache
from sw_utils import test_initial_condition_adjoint, test_gradient_array, pprint
from turbines import *
from mini_model import *
from dolfin import *
from dolfin_adjoint import *

# Global counter variable for vtk output
count = 0
# An animated plot to visualise the development of the functional value
plot = AnimatedPlot(xlabel='Iteration', ylabel='Functional value')

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  config = sw_config.DefaultConfiguration(nx=30, ny=10)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  pprint("Wave period (in h): ", period/60/60)
  config.params["dump_period"] = 1
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4
  config.params["dt"] = period/2
  config.params["finish_time"] = 5.*period/4 
  config.params["theta"] = 0.6
  config.params["include_advection"] = True 
  config.params["include_diffusion"] = True 
  config.params["diffusion_coef"] = 20.0
  config.params["newton_solver"] = False 
  config.params["picard_iterations"] = 20
  config.params['basename'] = "p2p1"
  config.params["run_benchmark"] = False 
  config.params['solver_exclude'] = ['cg', 'lu']
  info_green("Approximate CFL number (assuming a velocity of 2): " +str(2*config.params["dt"]/config.mesh.hmin())) 

  #set_log_level(DEBUG)
  set_log_level(20)
  #dolfin.parameters['optimize'] = True
  #dolfin.parameters['optimize_use_dofmap_cache'] = True
  #dolfin.parameters['optimize_use_tensor_cache'] = True
  #dolfin.parameters['form_compiler']['optimize'] = True
  dolfin.parameters['form_compiler']['cpp_optimize'] = True
  dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

  # Turbine settings
  config.params["quadratic_friction"] = True
  config.params["friction"] = 0.0025
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [] 
  border_x = 500
  border_y = 300
  for x_r in numpy.linspace(0.+border_x, config.params["basin_x"]-border_x, 3):
    for y_r in numpy.linspace(0.+border_y, config.params["basin_y"]-border_y, 2):
      config.params["turbine_pos"].append((float(x_r), float(y_r)))

  info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
  # Choosing a friction coefficient of > 0.02 ensures that overlapping turbines will lead to
  # less power output.
  config.params["turbine_friction"] = 0.2*numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 200
  config.params["turbine_y"] = 200

  return config

def initial_control(config):
  # We use the current turbine settings as the intial control
  res = numpy.reshape(config.params['turbine_pos'], -1).tolist()
  return numpy.array(res)

def j_and_dj(m):
  adj_reset()

  # Change the control variables to the config parameters
  config.params["turbine_pos"] = numpy.reshape(m, (-1, 2))

  debugging["record_all"] = True

  W = sw_lib.p2p1(config.mesh)

  state = Function(W, name="Current_state")
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U, name = 'friction')
  tfd = Function(U, name = 'friction_derivative')

  # Set up the turbine friction field using the provided control variable
  tf.interpolate(Turbines(config.params))

  global count
  count+=1
  sw_lib.save_to_file_scalar(tf, "turbines_t=."+str(count)+".x")

  # Scale the turbines in the functional for a physically consistent power/friction curve
  turbine_cache = build_turbine_cache(config.params, U, turbine_size_scaling=0.5)
  functional = DefaultFunctional(config.params, turbine_cache)

  # Solve the shallow water system
  j, djdm = sw_lib.sw_solve(W, config, state, turbine_field = tf, time_functional=functional)
  J = TimeFunctional(functional.Jt(state), static_variables = [turbine_cache["turbine_field"]], dt = config.params["dt"])
  #dJdfriction = compute_gradient(J, InitialConditionParameter("friction"))
  adj_state = sw_lib.adjoint(state, config.params, J, until={"name": "friction", "timestep": 0, "iteration": 0})

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
  j = j_and_dj_mem(m)[0]*10**-4
  pprint('Evaluating j(', m.__repr__(), ')=', j)
  plot.addPoint(j) 
  plot.savefig("plot_functional.png")
  return j 

def dj(m):
  dj = j_and_dj_mem(m)[1]*10**-4
  pprint('Evaluating dj(', m.__repr__(), ')=', dj)
  # Return the derivatives with respect to the position only
  return dj[len(config.params['turbine_friction']):]

config = default_config()
m0 = initial_control(config)

p = numpy.random.rand(len(m0))
minconv = test_gradient_array(j, dj, m0, seed=0.00001, perturbation_direction=p)
if minconv < 1.98:
  print "The gradient taylor remainder test failed."
  sys.exit(1)

g = lambda m: []
dg = lambda m: []

f = IPOptUtils.IPOptFunction()
# Overwrite the functional and gradient function with our implementation
f.objective= j 
f.gradient= dj 

# Get the upper and lower bounds for the turbine positions
lb, ub = IPOptUtils.position_constraints(config.params)

nlp = ipopt.problem(len(m0), 
                    0, 
                    f, 
                    numpy.array(lb),
                    numpy.array(ub))
nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption('tol', 1e-9)
nlp.addOption('print_level', 5)
nlp.addOption('check_derivatives_for_naninf', 'yes')
# A -1.0 scaling factor transforms the min problem to a max problem.
nlp.addOption('obj_scaling_factor', -1.0)
# Use an approximate Hessian since we do not have second order information.
nlp.addOption('hessian_approximation', 'limited-memory')
nlp.addOption('max_iter', 12)

m, info = nlp.solve(m0)
pprint(info['status_msg'])
pprint("Solution of the primal variables: m=%s\n" % repr(m))
pprint("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))
pprint("Objective=%s\n" % repr(info['obj_val']))

list_timings()
