''' Runs the forward model with a single turbine and prints some statistics '''
from opentidalfarm import *
set_log_level(INFO)

parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -fno-math-errno -march=native'        
parameters['form_compiler']['quadrature_degree'] = 20

basin_x = 640.
basin_y = 320.

inflow_direction = [1, 0]

config = SteadyConfiguration("mesh.xml", inflow_direction=inflow_direction)
config.functional = PowerCurveFunctional

# Place one turbine 
turbine_pos = [[basin_x/3-25, basin_y/2], 
               [basin_x/3+25, basin_y/2]] 

print0("Turbine position: " + str(turbine_pos))
config.set_turbine_pos(turbine_pos, friction=1.0)

u = 2.5

# Boundary conditions
bc = DirichletBCSet(config)
bc.add_constant_flow(1, u, direction=inflow_direction)
bc.add_zero_eta(2)
config.params['bctype'] = 'strong_dirichlet'
config.params['strong_bc'] = bc

rf = ReducedFunctional(config)
m = rf.initial_control()

seed = 1e-4
minconv = helpers.test_gradient_array(rf.j, rf.dj, m, seed=seed)

if minconv < 1.9:
    info_red("The gradient taylor remainder test failed.")
    sys.exit(1)
else:
    info_green("The gradient taylor remainder test passed.")
