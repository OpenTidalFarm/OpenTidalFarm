from opentidalfarm import *
set_log_level(INFO)

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = SteadyConfiguration("mesh.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params["controls"] = ["turbine_friction"]
config.params["turbine_parametrisation"] = "smooth"
config.params["automatic_scaling"] = False 

class Site(SubDomain):
    def inside(self, x, on_boundary):
	border = 10
        return (between(x[0], (site_x_start+border, site_x_start+site_x-border)) and 
				between(x[1], (site_y_start+border, site_y_start+site_y-border)))

site = Site()
domains = CellFunction("size_t", config.domain.mesh)
domains.set_all(0)
site.mark(domains, 1)
config.site_dx = Measure("dx")[domains]

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)

config.info()

rf = ReducedFunctional(config)

maximize(rf, method = "L-BFGS-B", options={'maxiter':100}, bounds=[0, 1])
