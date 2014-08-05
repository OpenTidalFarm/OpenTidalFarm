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
config = SteadyConfiguration("mesh_coarse.xml", inflow_direction=inflow_direction)
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

config.params['initial_condition'] = ConstantFlowInitialCondition(config, val=[1, 1, 1])
config.params['viscosity'] = 12

# Work out the expected delta eta for a free-stream of 2.5 m/s (without turbines) 
u_free_stream = 2.5
print "Target free-stream velocity (without turbines): ", u_free_stream
# Assume balance between pressure and friction term

delta_eta = config.params["friction"](())/config.params["depth"]/config.params["g"]
delta_eta *= u_free_stream**2
delta_eta *= basin_x
print "Derived head-loss difference to achieve target free-stream: ", delta_eta

# Set Boundary conditions
bc = DirichletBCSet(config)
#expression_left = Expression("delta_eta/2*cos(pi*(t-1))", delta_eta=delta_eta, t=0)
#expression_right = Expression("-delta_eta/2*cos(pi*(t-1))", delta_eta=delta_eta, t=0)
expression_left = Expression("delta_eta/2+0*t", delta_eta=delta_eta, t=0)
expression_right = Expression("-delta_eta/2+0*t", delta_eta=delta_eta, t=0)
bc.add_analytic_eta(1, expression_left)
bc.add_analytic_eta(2, expression_right)
config.params['strong_bc'] = bc

# Place some turbines 
deploy_turbines(config, nx=8, ny=4)
config.info()

power = PowerFunctional(config)
objective = power

rf = ReducedFunctional(config, functional = objective)
m0 = rf.initial_control()
rf.j(m0)
rf.dj(m0, forget=False)
import sys; sys.exit()

p = numpy.random.rand(len(m0))
seed = 0.1
minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=seed, perturbation_direction=p)

print "minconv", minconv
