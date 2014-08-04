''' Runs the forward model with a single turbine and prints some statistics '''
from opentidalfarm import *
from opentidalfarm.helpers import function_eval
import matplotlib.pyplot as plt
import numpy
set_log_level(INFO)
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -fno-math-errno -march=native'        
parameters['form_compiler']['quadrature_degree'] = 20

basin_x = 640.
basin_y = 320.

inflow_direction = [1, 0]

config = SteadyConfiguration("mesh.xml", inflow_direction=inflow_direction)
config.functional = PowerCurveFunctional
#config.params['implicit_turbine_thrust_parametrisation'] = True
config.params['turbine_thrust_parametrisation'] = True
config.params['initial_condition'] = ConstantFlowInitialCondition(config)
config.params['automatic_scaling'] = False
#config.params['viscosity'] = 10.

# Place one turbine 
turbine_pos = [[basin_x/3, basin_y/2]]

print0("Turbine position: " + str(turbine_pos))
config.set_turbine_pos(turbine_pos, friction=1.0)

def sw_callback(config, state):
    ux = state[0]
    uy = state[1]
    eta = state[2]
    up_u = state[3]

    print0("Inflow velocity: ", function_eval(ux, (10, 160)))
    print0("Estimated upstream velocity: ", function_eval(up_u, (640. / 3, 160)))

config.params["postsolver_callback"] = sw_callback
config.info()

fac = 1.5e6/(3**3) # Scaling factor such that for 3 m/s, the turbine produces 1.5 MW
us = numpy.linspace(0, 5, 6)
powers = []
for u in us:  
    # Boundary conditions
    bc = DirichletBCSet(config)
    bc.add_constant_flow(1, u, direction=inflow_direction)
    bc.add_analytic_eta(2, 0.0)
    config.params['bctype'] = 'strong_dirichlet'
    config.params['strong_bc'] = bc

    model = ReducedFunctional(config)
    m = model.initial_control()
    j, state = model.compute_functional_mem(m, return_final_state = True)

    print0("Extracted power (MW): %f" % (j*10**-6))
    print0("Expected power (MW): %f" % (min(1.5e6, fac*u**3)*10**-6))
    powers.append(j)

plt.clf()
plt.plot(us, [p*1e-6 for p in powers], label="Approximated")
plt.plot(us, [min(1.5e6, fac*u**3)*10**-6 for u in us], label="Analytical")
plt.legend(loc=2)

plt.savefig("power_plot.pdf", format='pdf')
