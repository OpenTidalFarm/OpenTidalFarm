''' Runs the forward model with a single turbine and prints some statistics '''
from opentidalfarm import *
import matplotlib.pyplot as plt
set_log_level(ERROR)

basin_x = 640.
basin_y = 320.

inflow_direction = [1, 0]

config = SteadyConfiguration("mesh.xml", inflow_direction=inflow_direction)
config.functional = PowerCurveFunctional
config.params['automatic_scaling'] = False

# Place one turbine 
turbine_pos = [[basin_x/3, basin_y/2], 
               [basin_x/3 + 50, basin_y/2]] 

print0("Turbine position: " + str(turbine_pos))
config.set_turbine_pos(turbine_pos)

us = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
powers = []
for u in us:  
    # Boundary conditions
    bc = DirichletBCSet(config)
    bc.add_constant_flow(1, u, direction=inflow_direction)
    bc.add_zero_eta(2)
    config.params['bctype'] = 'strong_dirichlet'
    config.params['strong_bc'] = bc

    model = ReducedFunctional(config)
    m = model.initial_control()
    j, state = model.compute_functional_mem(m, return_final_state = True)
    print0("Extracted Power (MW): %f" % (j*10**-6))
    powers.append(j)

plt.plot(us, powers)
plt.savefig("power_plot.pdf", format='pdf')
