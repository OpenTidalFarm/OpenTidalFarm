from opentidalfarm import *
set_log_level(INFO)

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2
site_y_start = (basin_y - site_y)/2

# Define a turbine farm
farm = RectangularFarm(site_x_start, site_x_start + site_x,
                       site_y_start, site_y_start + site_y)

# Define a turbine to use
turbine = Turbine(diameter=20.0, minimum_distance=40.0, friction=21.0)

config = SteadyConfiguration("mesh_coarse.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place some turbines
farm.add_regular_turbine_layout(turbine, num_x=8, num_y=4)

# New API is not yet complete, so do this in the meantime.
m = farm.serialize(turbine_pos=True, turbine_friction=True)
turbine_friction = m[:len(m)/3]
positions = m[len(m)/3:]
turbine_pos = []
for i in range(len(positions)/2):
    turbine_pos.append((positions[2*i], positions[2*i+1]))

config.params["turbine_pos"] = turbine_pos
config.params["turbine_friction"] = turbine_friction

config.info()

rf = ReducedFunctional(config)

lb, ub = farm.site_boundary_constraints()
ineq = farm.minimum_distance_constraints(True,False)
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP", options={'maxiter':3})
