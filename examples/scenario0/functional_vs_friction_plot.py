''' An example that plots the power ouput of a single turbine for different fricition values. (see also test_optimal_friction_single_turbine) '''
import sys
import matplotlib.pyplot as plt
from opentidalfarm import *
set_log_level(ERROR)

basin_x = 640.
basin_y = 320.

config = SteadyConfiguration("mesh.xml", inflow_direction = [1, 0])
config.params['automatic_scaling'] = False

# Place one turbine 
offset = 0.0
turbine_pos = [[basin_x/3 + offset, basin_y/2 + offset]] 
info_green("Turbine position: " + str(turbine_pos))
config.set_turbine_pos(turbine_pos)
config.params['controls'] = ['turbine_friction']

config.info()

model = ReducedFunctional(config, scale=10**-6)
m0 = model.initial_control()
m_list = [numpy.ones(len(m0))*i for i in numpy.linspace(0, 50, 11)] + [numpy.ones(len(m0))*i for i in numpy.linspace(60, 200, 8)]
info_green('Testing friction coefficients: ' + str([x[0] for x in m_list]))

# We already know that a zero friction leads to a zero power 
P = [0]
f = [0]
for m in m_list: 
  P.append(model.j(m))
  f.append(m[0])

# Plot the results
if MPI.process_number() == 0:
  info_green('The maximum functional value of ' + str(max(P)) + ' is achieved with a friction coefficient of ' + str(f[numpy.argmax(P)]) + '.')

  scaling = 0.7
  plt.figure(1, figsize = (scaling*7., scaling*4.))
  plt.gcf().subplots_adjust(bottom=0.15)
  plt.plot(f, P, color = "black")
  plt.ylabel('Power output [MW]')
  plt.xlabel('Friction coefficient K')
  plt.yticks(numpy.arange(0, 2.5, 0.5))
  plt.savefig('turbine_friction_vs_power.pdf')
