from opentidalfarm import *
from common import TidalForcing, BathymetryDepthExpression
from math import pi
import os.path
forward_only = True
test_gradient = True

config = UnsteadyConfiguration("mesh/coast_idBoundary_utm.xml", [1, 1]) 
config.params['initial_condition'] = ConstantFlowInitialCondition(config) 
config.params['diffusion_coef'] = Constant(250.0)
config.params["controls"] = ["turbine_friction"]
config.params["turbine_parametrisation"] = "smooth"
config.params["automatic_scaling"] = False 
config.params['friction'] = Constant(0.0025)
config.params['cache_forward_state'] = True
config.params['base_path'] = "results_multisteady_masked"

# Perform only two timesteps
config.params['include_time_term'] = False
config.params['start_time'] = -1 * 16 * 600 
config.params['dt'] = 2 * 16 * 600 
config.params['finish_time'] = 3 * 16 * 600 
config.params['theta'] = 1.0 
config.params['functional_quadrature_degree'] = 0

# Tidal boundary forcing
bc = DirichletBCSet(config)

eta_expr = TidalForcing() 
bc.add_analytic_eta(1, eta_expr)
bc.add_analytic_eta(2, eta_expr)
# comment out if you want free slip:
bc.add_noslip_u(3)
config.params['bctype'] = 'strong_dirichlet'
config.params['strong_bc'] = bc

V_cg1 = FunctionSpace(config.domain.mesh, "CG", 1)
V_dg0 = FunctionSpace(config.domain.mesh, 'DG', 0)

# Bathymetry
bexpr = BathymetryDepthExpression('bathymetry.nc')
depth = interpolate(bexpr, V_cg1) 
depth_pvd = File(os.path.join(config.params["base_path"], "bathymetry.pvd"))
depth_pvd << depth

config.params['depth'] = depth
config.turbine_function_space = V_dg0 

domains = MeshFunction("size_t", config.domain.mesh, "mesh/coast_idBoundary_utm_physical_region.xml")

# Compute gradient of the batyhmetry
depth_grad = Function(V_dg0)
F = inner(TrialFunction(V_dg0) - (depth.dx(0)**2 + depth.dx(1)**2)**0.5, TestFunction(V_dg0))*dx
solve(lhs(F) == rhs(F), depth_grad)

# The maximum gradient in which turbines can be deployed
grad_limit = 0.015
jump = lambda x: 1 if x < grad_limit else 0 
vjump = numpy.vectorize(jump)

indic_arr = vjump(depth_grad.vector().get_local())
indic_arr *= domains.array() 

domains.set_values(numpy.array(indic_arr, dtype=numpy.uintp))
#plot(domains, interactive=True)
config.site_dx = Measure("dx")[domains]
f = File(os.path.join(config.params["base_path"], "turbine_farms.pvd"))
f << domains

config.params["save_checkpoints"] = True
config.info()

rf = ReducedFunctional(config, scale=-1e-6)
rf.load_checkpoint()

if forward_only or test_gradient:
    print "Running forward model"
    m0 = rf.initial_control()
    j = rf.j(m0, annotate=test_gradient)
    print "Power: ", j
    if test_gradient:
        dj = rf.dj_with_check(m0, forget=False, seed=0.01)

else:
  # The maximum friction is given by:
  # c_B = c_T*A_Cross / (2*A) = 0.6*pi*D**2/(2*9D**2) 
  max_ct = 0.6*pi/2/9
  print "Maximum turbine friction: %f." % max_ct

  rf = ReducedFunctional(config, scale=-1e-6)
  m_opt = maximize(rf, bounds = [0, max_ct], options = {"maxiter": 300})
