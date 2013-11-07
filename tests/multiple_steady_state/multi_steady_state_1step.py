from opentidalfarm import *
set_log_level(INFO)

inflow_direction = [1, 0]
# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = UnsteadyConfiguration("mesh_coarse.xml", inflow_direction=inflow_direction)
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Change the parameters such that in fact two steady state problems are solved consecutively
config.params['initial_condition'] = ConstantFlowInitialCondition(config, val=[1, 1, 1])
config.params['theta'] = 1
config.params['start_time'] = 0 
config.params['dt'] = 1 
config.params['finish_time'] = 1 
config.params['include_time_term'] = False
config.params['diffusion_coef'] = 16
config.params['functional_quadrature_degree'] = 0
config.params["newton_solver"] = True
config.params['k'] = pi/basin_x
#config.params["linear_solver"] = "umfpack"

# Work out the expected delta eta for a free-stream of 2.5 m/s (without turbines) 
# by assuming balance between the pressure and friction terms
u_free_stream = 2.5
print "Target free-stream velocity (without turbines): ", u_free_stream
delta_eta = config.params["friction"](())/config.params["depth"]/config.params["g"]
if config.params["quadratic_friction"]: 
	delta_eta *= u_free_stream**2
else:
	delta_eta *= u_free_stream
delta_eta *= basin_x
print "Derived head-loss difference to achieve target free-stream: ", delta_eta

# Set Boundary conditions
bc = DirichletBCSet(config)
expl = Expression("delta_eta/2*cos(pi*(t-1))", delta_eta=delta_eta, t=0)
expr = Expression("-delta_eta/2*cos(pi*(t-1))", delta_eta=delta_eta, t=0)
bc.add_analytic_eta(1, expl)
bc.add_analytic_eta(2, expr)
config.params['strong_bc'] = bc

# Place some turbines 
deploy_turbines(config, nx=8, ny=4)
config.info()

rf = ReducedFunctional(config)
m0 = rf.initial_control()
p = numpy.random.rand(len(m0))
seed = 0.1
minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=seed, perturbation_direction=p)

assert minconv > 1.9
