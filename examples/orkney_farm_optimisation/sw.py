from opentidalfarm import *
import ipopt

config = SteadyConfiguration("mesh/coast_idBoundary_utm.xml", inflow_direction=[0.9865837220518425, -0.16325611591095968]) 
config.params['diffusion_coef'] = 180.0
config.params["controls"] = ["turbine_friction"]
config.params["turbine_parametrisation"] = "smooth"
config.params["automatic_scaling"] = False 

bc = DirichletBCSet(self)
bc.add_constant_flow(1, 1.0, direction=inflow_direction)
bc.add_zero_eta(2)
self.params['bctype'] = 'strong_dirichlet'
self.params['strong_bc'] = bc

config.turbine_function_space = FunctionSpace(config.domain.mesh, 'DG', 0)

domains = MeshFunction("size_t", config.domain.mesh, "mesh/coast_idBoundary_utm_physical_region.xml")
#plot(domains, interactive=True)
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
