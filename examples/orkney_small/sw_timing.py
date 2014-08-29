from opentidalfarm import *
set_log_level(PROGRESS)

# Some domain information extracted from the geo file
site_x = 1000.
site_y = 500.
site_x_start = 1.03068e+07 
site_y_start = 6.52246e+06 - site_y 

inflow_x = 8400.
inflow_y = -1390.
inflow_norm = (inflow_x**2 + inflow_y**2)**0.5
inflow_direction = [inflow_x/inflow_norm, inflow_y/inflow_norm]
print "inflow_direction: ", inflow_direction

config = SteadyConfiguration("mesh/earth_orkney_converted.xml", inflow_direction = inflow_direction) 
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params['viscosity'] = 90.0
config.params["turbine_x"] = 40.
config.params["turbine_y"] = 40.
config.params["dump_period"] = 0

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)
config.params["turbine_friction"] = 0.5*numpy.array(config.params["turbine_friction"]) 

model = ReducedFunctional(config, scale = -1)
m0 = model.initial_control()

# Build the turbine cache 
config.turbine_cache.update(config)

time_forward = []
time_adjoint = []
for i in range(5):
	print "Running forward model round ", i
	t1 = Timer("Forward model")
	model.run_forward_model_mem.fn(m0) 
	time_forward.append(t1.stop())
	print "Forward model runtime: ", time_forward[-1]

	from dolfin_adjoint import replay_dolfin
	print "Replaying model round ", i
	t11 = Timer("Replay forward model")
	replay_dolfin()
	print "Replay model runtime: ", t11.stop() 

	t2 = Timer("Adjoint model")
	model.run_adjoint_model_mem.fn(m0) 
	time_adjoint.append(t2.stop())
	print "Adjoint model runtime: ", time_adjoint[-1]

print "Smallest runtime for forward model: ", min(time_forward)
print "Smallest runtime for forward + adjoint model: ", min(time_forward) + min(time_adjoint)
print "Ratio: ", 1.0 + min(time_adjoint) / min(time_forward)
