''' Runs the forward model with a single turbine and prints some statistics '''
from opentidalfarm import *
from dolfin import log, INFO, ERROR
set_log_level(INFO)

parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -fno-math-errno \
        -march=native'
parameters['form_compiler']['quadrature_degree'] = 4

basin_x = 640.
basin_y = 320.

inflow_direction = [1, 0]

config = SteadyConfiguration("mesh.xml", inflow_direction=inflow_direction)
config.functional = PowerCurveFunctional
config.params['turbine_thrust_parametrisation'] = True
config.params['initial_condition'] = ConstantFlowInitialCondition(config)
k = pi/basin_x
config.params["flather_bc_expr"] = Expression(
    ("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"),
    eta0=2.,
    g=config.params["g"],
    depth=config.params["depth"],
    t=config.params["current_time"],
    k=k
)

# Place one turbine
turbine_pos = [[basin_x/3-25, basin_y/2],
               [basin_x/3+25, basin_y/2]]

print0("Turbine position: " + str(turbine_pos))
config.set_turbine_pos(turbine_pos, friction=1.0)

u = 2.5

# Boundary conditions
bc = DirichletBCSet(config)
bc.add_constant_flow(1, u, direction=inflow_direction)
bc.add_analytic_eta(2, 0.0)
config.params['bctype'] = 'strong_dirichlet'
config.params['strong_bc'] = bc

rf = ReducedFunctional(config)
m = rf.initial_control()

seed = 1e-4
minconv = helpers.test_gradient_array(rf.j, rf.dj, m, seed=seed)

if minconv < 1.9:
    log(ERROR, "The gradient taylor remainder test failed.")
    sys.exit(1)
else:
    log(INFO, "The gradient taylor remainder test passed.")
