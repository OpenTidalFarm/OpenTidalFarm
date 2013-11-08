''' This example optimises the position of three turbines using the hallow water model. '''

import sys
from opentidalfarm import *
import opentidalfarm.domains
set_log_level(ERROR)

config = DefaultConfiguration(nx=5, ny=5)
config.set_domain(opentidalfarm.domains.RectangularDomain(3000, 1000, 5, 5))
config.params['finish_time'] = config.params["start_time"] + 10*config.params["dt"]

# Switch to a smooth turbine representation
config.params["controls"] = ["turbine_friction"]
config.params["turbine_parametrisation"] = "smooth"

site_x_start = 750
site_x = 1500
site_y_start = 250
site_y = 500

k = pi/site_x
config.params["flather_bc_expr"] = Expression(("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"), 
                                 eta0=2., 
                                 g=config.params["g"], 
                                 depth=config.params["depth"], 
                                 t=config.params["current_time"], 
                                 k=k)

class Site(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (site_x_start, site_x_start+site_x)) and
            between(x[1], (site_y_start, site_y_start+site_y)))

site = Site()
domains = CellFunction("size_t", config.domain.mesh)
domains.set_all(0)
site.mark(domains, 1)
config.site_dx = Measure("dx")[domains]

rf = ReducedFunctional(config)
# Ensure the same seed value accross all CPUs
numpy.random.seed(33)
m0 = numpy.random.rand(len(rf.initial_control()))

seed = 0.1 
minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=seed)

if minconv < 1.9:
    info_red("The gradient taylor remainder test failed.")
    sys.exit(1)
else:
    info_green("The gradient taylor remainder test passed.")
