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
m_list = [m0*i for i in numpy.linspace(.5, 2., 8)]
info_green('Testing friction coefficients: ' + str(m_list))

# We already know that a zero friction leads to a zero power 
P = [0]
f = [0]
for m in m_list: 
  P.append(model.j(m, forward_only = True))
  f.append(m[0])

# Plot the results
if MPI.process_number() == 0:
  plt.figure(1)
  plt.plot(f, P)
  info_green('The maximum functional value of ' + str(max(P)) + ' is achieved with a friction coefficient of ' + str(f[numpy.argmax(P)]) + '.')
  plt.ylabel('Power output')
  plt.xlabel('Turbine coefficient')
  plt.savefig('example_single_turbine_friction_vs_power_plot.pdf')
  plt.show()
  plt.hold()
