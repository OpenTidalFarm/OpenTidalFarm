from opentidalfarm import *
import datetime
utm_zone = 30
utm_band = 'V'

config = UnsteadyConfiguration("mesh/coast_idBoundary_utm.xml", [1, 1]) 
config.params['initial_condition'] = ConstantFlowInitialCondition(config) 
config.params['diffusion_coef'] = 180.0
config.params["controls"] = ["turbine_friction"]
config.params["turbine_parametrisation"] = "smeared"
config.params["automatic_scaling"] = False 
config.params['friction'] = Constant(0.0025)
config.params['base_path'] = "forward_only"

config.params['start_time'] = 0 
config.params['dt'] = 600 
config.params['finish_time'] = 12.5*60*60 
config.params['theta'] = 1.0 

# Tidal boundary forcing
bc = DirichletBCSet(config)
eta_expr = TidalForcing(grid_file_name='netcdf/gridES2008.nc',
						data_file_name='netcdf/hf.ES2008.nc',
						ranges=((-4.0,0.0), (58.0,61.0)),
						utm_zone=utm_zone, 
		                utm_band=utm_band, 
						initial_time=datetime.datetime(2001, 9, 18, 0),
						constituents=['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2'])

bc.add_analytic_eta(1, eta_expr)
bc.add_analytic_eta(2, eta_expr)
# comment out if you want free slip:
#bc.add_noslip_u(3)
config.params['bctype'] = 'strong_dirichlet'
config.params['strong_bc'] = bc

# Bathymetry
bexpr = BathymetryDepthExpression('netcdf/bathymetry.nc', utm_zone=utm_zone, utm_band=utm_band)
depth = interpolate(bexpr, FunctionSpace(config.domain.mesh, "CG", 1))
depth_pvd = File(config.params["base_path"] + "/bathymetry.pvd")
depth_pvd << depth

config.params['depth'] = depth

config.turbine_function_space = FunctionSpace(config.domain.mesh, 'DG', 0)

domains = MeshFunction("size_t", config.domain.mesh, "mesh/coast_idBoundary_utm_physical_region.xml")
#plot(domains, interactive=True)
config.site_dx = Measure("dx")[domains]

config.info()

rf = ReducedFunctional(config, scale=-1e-6)

print "Running forward model"
m0 = rf.initial_control()
rf.j(m0, annotate=False)
print "Finished"
import sys; sys.exit(1)

#maximize(rf, bounds=(0, 1))

#class Problem(object):
#    def __init__(self, rf):
#        self.rf = rf
#
#    def objective(self, x):
#        return self.rf.j(x)
#
#    def gradient(self, x):
#        return self.rf.dj(x, forget=False)
#
#    def constraints(self, x):
#        return [sum(x)]
#
#    def jacobian(self, x):
#        return [[1]*len(x)]
#
#
#
#nlp = ipopt.problem(n=len(m0), 
#                    m=0, 
#                    problem_obj=Problem(rf),
#                    lb=[0]*len(m0),
#                    ub=[1]*len(m0),
#                    cl=[0],
#                    cu=[1000])
#
#nlp.addOption("hessian_approximation", "limited-memory")
#nlp.solve(m0)
