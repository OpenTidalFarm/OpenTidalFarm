from opentidalfarm import *
set_log_level(INFO)

# Some domain information extracted from the geo file
basin_x = 1280.
basin_y = 640.
site_x = 640.
site_y = 320.
site_x_start = (basin_x - site_x)/2
site_y_start = (basin_y - site_y)/2
config = SteadyConfiguration("mesh_coarse.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

tempconfig= SteadyConfiguration("mesh_coarse.xml", inflow_direction = [1, 0])
tempconfig.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start,
        site_y_start + site_y)


class Bathymetry(Expression):
    def __init__(self, config, max_depth, min_depth):
        self.config = config
        self.step = (config.domain.site_y_start + config.domain.site_y_end)/3
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.m = (self.max_depth-self.min_depth)/self.step
        self.m2 = (self.min_depth- self.max_depth)/self.step

    def eval(self, values, x):
        if x[1] > 2 * self.step:
            values[:] = self.m2 * (x[1] - 2 * self.step) + self.max_depth + sin(100 * x[0]) + 2*sin(0.5 * x[0]) + 0.5*sin(50 * x[1]) + sin(0.5 * x[1])
        elif x[1] > self.step and x[1] <= 2 * self.step:
            values[:] = self.max_depth +  sin(100 * x[0]) + 2*sin(0.5 * x[0]) + 0.5*sin(50 * x[1]) + sin(0.5 * x[1])
        else:
            values[:] = (self.m * x[1] + self.min_depth) + sin(100 * x[0]) + 2*sin(0.5 * x[0]) + 0.5*sin(50 * x[1]) + sin(0.5 * x[1])


bathymetry = Bathymetry(config, 50., 40.)
depth = interpolate(bathymetry, FunctionSpace(config.domain.mesh, "CG", 1))

bathymetry2 = Bathymetry(tempconfig, 50., 40.)
depth2 =  interpolate(bathymetry2, FunctionSpace(tempconfig.domain.mesh, "CG", 1))

tempconfig.params['depth'] = depth2
temprf = ReducedFunctional(tempconfig)
config.params['ambient_flow_field'] = temprf.get_ambient_flow_field(tempconfig)

config.params['ecology'] = True

# Place some turbines
deploy_turbines(config, nx = 8, ny = 4)
config.params["depth"] = depth
config.info()

rf = ReducedFunctional(config)

lb, ub = position_constraints(config)
ineq = get_minimum_distance_constraint_func(config)
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP", options={'maxiter':3})
