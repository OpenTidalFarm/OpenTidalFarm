''' Runs the forward model with a single turbine and prints some statistics '''
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

config.info()

model = ReducedFunctional(config)
m = model.initial_control()
j, state = model.compute_functional_mem(m, return_final_state = True)
info_green("Extracted Power (MW): %f" % (j*10**-6))

# Compute the Lanchester-Betz limit.
# The notation follows C. Garretta and P. Cummins, Limits to tidal current power
try: 
    u0 = state((1., basin_y/2))[0]
except RuntimeError:
    u0 = -100000
    pass
u0 = MPI.max(u0)


tf = config.turbine_cache.cache['turbine_field']

# Computed the averaged velocity at the turbine, weighted with the amount of friction
tf_vol = assemble(inner(tf, Constant(1))*dx)
u1_weighted = assemble(inner(state[0]*tf, Constant(1))*dx) / tf_vol

# Computed the averaged velocity at the turbine
indicator = conditional(gt(tf, 0), 1.0, 0.0)
indicator_func = project(indicator, FunctionSpace(tf.function_space().mesh(), "DG", 0))
indicator_vol = assemble(inner(indicator_func, Constant(1))*dx)
u1_indicator = assemble(inner(state[0]*indicator_func, Constant(1))*dx) / indicator_vol

# Compute the average velocity in the turbine area
#u1 = 0.0
#for o_x in config.params["turbine_x"]*numpy.linspace(-1, 1, 50)/2:
#    for o_y in config.params["turbine_y"]*numpy.linspace(-1, 1, 50)/2:
#        try: 
#            u1 += state(numpy.array(turbine_pos[0]) + [o_x, o_y])[0]
#        except RuntimeError:
#            pass
#u1 = MPI.sum(u1)/(50*50)
try: 
    u1 = state(numpy.array(turbine_pos[0]))[0]
except RuntimeError:
    u1 = -100000
    pass
u1 = MPI.max(u1)

info_green("Inflow velocity u0: " + str(u0) + " m/s")
info_green("-"*80)
info_green("Turbine velocity u1 (evaluated at the turbine center): " + str(u1) + " m/s")
info_green("u1/u0: " + str(u1/u0) + " (Lanchester-Betz limit is reached at u1/u0 = 0.66666)")
info_green("-"*80)
info_green("Turbine velocity u1 (averaged over the turbine area): " + str(u1_indicator) + " m/s")
info_green("u1/u0: " + str(u1_indicator/u0) + " (Lanchester-Betz limit is reached at u1/u0 = 0.66666)")
info_green("-"*80)
info_green("Turbine velocity u1 (weighted averaged over the turbine area): " + str(u1_weighted) + " m/s")
info_green("u1/u0: " + str(u1_weighted/u0) + " (Lanchester-Betz limit is reached at u1/u0 = 0.66666)")

dj = model.dj(m, forget=True)
info_green("Functional gradient: " + str(dj))
