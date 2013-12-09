from opentidalfarm import *
from common import TidalForcing, BathymetryDepthExpression
import distance
from math import pi
import os.path
forward_only = True
test_gradient = True

config = UnsteadyConfiguration("mesh/orkney.xml", [1, 1]) 
config.params['initial_condition'] = ConstantFlowInitialCondition(config) 
config.params['diffusion_coef'] = Constant(250.0)
config.params["controls"] = ["turbine_friction"]
config.params["turbine_parametrisation"] = "smeared"
config.params["automatic_scaling"] = False 
config.params['friction'] = Constant(0.0025)
config.params['cache_forward_state'] = True
config.params['base_path'] = "results_multisteady_masked"

# Perform only two timesteps
config.params['include_time_term'] = True
config.params['start_time'] = 0
config.params['dt'] = 3600.
config.params['finish_time'] = 48.*3600.
config.params['theta'] = 1.0 
config.params['functional_quadrature_degree'] = 0

#config.params['newton_solver'] = False
#config.params['picard_iterations'] = 2

# Tidal boundary forcing
bc = DirichletBCSet(config)

eta_expr = TidalForcing() 
for boundary_id in range(2,7):
  bc.add_analytic_eta(boundary_id, eta_expr)
# comment out if you want free slip:
bc.add_noslip_u(1)
config.params['bctype'] = 'strong_dirichlet'
config.params['strong_bc'] = bc

V_cg1 = FunctionSpace(config.domain.mesh, "CG", 1)
V_dg0 = FunctionSpace(config.domain.mesh, 'DG', 0)

# Bathymetry
bexpr = BathymetryDepthExpression('bathymetry.nc')
#bexpr = Constant(50.0)
depth = interpolate(bexpr, V_cg1) 
depth_pvd = File(os.path.join(config.params["base_path"], "bathymetry.pvd"))
depth_pvd << depth

config.params['depth'] = depth
config.turbine_function_space = V_dg0 

cell_depth = interpolate(depth, V_dg0).vector().get_local()
cell_x = interpolate(Expression("x[0]"), V_dg0).vector().get_local()
cell_y = interpolate(Expression("x[1]"), V_dg0).vector().get_local()
cell_distance_to_boundary = interpolate(distance.DistanceToCoast(config), V_dg0).vector().get_local()

# where to allow turbines:
min_depth = 20.0
max_depth = 1000.0 # this is not reached, so no maximum is applied
wh  = (min_depth < cell_depth) & (cell_depth < max_depth) & (cell_distance_to_boundary<10*1.e3) & \
                                                            (cell_x > 468000.) & (cell_y > 6480000.)
domains = MeshFunction('size_t', config.domain.mesh, 2)
domains.set_values(numpy.array(wh, dtype=numpy.uintp))
#plot(domains, interactive=True)
config.site_dx = Measure("dx")[domains]
f = File(os.path.join(config.params["base_path"], "turbine_farms.pvd"))
f << domains

config.params["save_checkpoints"] = True
config.info()

rf = ReducedFunctional(config, scale=-1e-6)
#rf.load_checkpoint()

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
  m_opt = maximize(rf, bounds = [0, max_ct], options = {"maxiter": 300})
