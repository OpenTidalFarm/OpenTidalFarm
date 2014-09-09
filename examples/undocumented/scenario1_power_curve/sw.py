from opentidalfarm import *
set_log_level(INFO)
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -fno-math-errno -march=native'        
parameters['form_compiler']['quadrature_degree'] = 20

basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 

inflow_direction = [1, 0]

config = SteadyConfiguration("mesh.xml", inflow_direction=inflow_direction)
config.functional = PowerCurveFunctional
config.params['turbine_thrust_parametrisation'] = True
config.params['initial_condition'] = ConstantFlowInitialCondition(config)
config.params['viscosity'] = 30.

config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)

bc = DirichletBCSet(config)
bc.add_constant_flow(1, 2.0, direction=inflow_direction)
bc.add_analytic_eta(2, 0.0)
config.params['bctype'] = 'strong_dirichlet'
config.params['strong_bc'] = bc

config.info()

rf = ReducedFunctional(config)

lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config, min_distance=5)
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP", options = {"maxiter": 200})
