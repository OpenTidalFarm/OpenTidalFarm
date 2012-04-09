''' An example that plots the power ouput of a single turbine for different fricition values. (see also test_optimal_friction_single_turbine) '''
import sys
import configuration 
import numpy
import matplotlib.pyplot as plt
from dolfin import *
from dolfin_adjoint import *
from reduced_functional import ReducedFunctional

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
config = configuration.PaperConfiguration(nx = 40, ny = 20)
config.params['dump_period'] = 1

# Turbine settings
config.params['turbine_pos'] = [[config.params['basin_x']/2, config.params['basin_y']/2]]
config.params['turbine_friction'] = numpy.ones(len(config.params['turbine_pos']))
config.params['controls'] = ['turbine_friction']
config.params['finish_time'] = 1.5/4*config.period

# Set up the model
model = ReducedFunctional(config)
m0 = model.initial_control()
m_list = [m0*i for i in numpy.linspace(1., 10., 10)]
info_green('Testing friction coefficients: ' + str(m_list))

P = [0*m0, 0.]
for m in m_list: 
  P.append(model.j(m, forward_only = True))

# Plot the results
if MPI.process_number() == 0:
  plt.figure(1)
  plt.plot(f, P)
  info_green('The maximum functional value of ' + str(max(P)) + ' is achieved with a friction coefficient of ' + str(f[numpy.argmax(P)]) + '.')
  plt.title('Power output of a single turbine with quadratic friction.')
  plt.ylabel('Power output')
  plt.xlabel('Friction coefficient')
  plt.show()
  plt.hold()
