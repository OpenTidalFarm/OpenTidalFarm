from opentidalfarm import *
import ipopt
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
config.params["turbine_parametrisation"] = "smeared"
config.params["automatic_scaling"] = False 

config.turbine_function_space = FunctionSpace(config.domain.mesh, 'DG', 0)

# Define the site domain
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

config.info()

rf = ReducedFunctional(config, scale=-1e-6)

class Problem(object):
    def __init__(self, rf):
	self.rf = rf

    def objective(self, x):
	return self.rf.j(x)

    def gradient(self, x):
	return self.rf.dj(x, forget=False)

    def constraints(self, x):
	return [sum(x)]

    def jacobian(self, x):
	return [[1]*len(x)]


m0 = rf.initial_control()
nlp = ipopt.problem(n=len(m0), 
	            m=0, 
		    problem_obj=Problem(rf),
		    lb=[0]*len(m0),
		    ub=[1]*len(m0),
		    cl=[0],
		    cu=[1000])

nlp.addOption("hessian_approximation", "limited-memory")
nlp.solve(m0)
