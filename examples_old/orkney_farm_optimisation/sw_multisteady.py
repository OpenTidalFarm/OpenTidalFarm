from opentidalfarm import *
import datetime
from math import pi
import os.path
import sys
forward_only = False
test_gradient = False
utm_zone = 30
utm_band = 'V'
# If farmselector is None, all farms are optimised. 
# If farmselector is between 1 and 4, only the selected farm is optimised
if len(sys.argv) > 1: 
    farm_selector = int(sys.argv[1])
else:
    farm_selector = None  

if farm_selector is None:
    print "Optimising all farms."
    mesh_basefile = "mesh/coast_idBoundary_utm_no_islands"
else:
    print "Optimising farm %i only." % farm_selector
    mesh_basefile = "mesh/coast_idBoundary_utm_no_islands_individual_farm_ids"

config = UnsteadyConfiguration(mesh_basefile + ".xml", [1, 1]) 
config.params['initial_condition'] = ConstantFlowInitialCondition(config) 
config.params['viscosity'] = Constant(250.0)
config.params["controls"] = ["turbine_friction"]
config.params["turbine_parametrisation"] = "smeared"
config.params["automatic_scaling"] = False 
config.params['friction'] = Constant(0.0025)
config.params['cache_forward_state'] = True
if farm_selector is None:
    config.params['base_path'] = "results_multisteady"
else:
    config.params['base_path'] = "results_multisteady_farm_%i_only" % farm_selector

# Perform only two timesteps
config.params['include_time_term'] = False
config.params['start_time'] = -1 * 16 * 600 
config.params['dt'] = 2 * 16 * 600 
config.params['finish_time'] = 3 * 16 * 600 
config.params['theta'] = 1.0 
config.params['functional_quadrature_degree'] = 0

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
bc.add_noslip_u(3)
config.params['bctype'] = 'strong_dirichlet'
config.params['strong_bc'] = bc

V_cg1 = FunctionSpace(config.domain.mesh, "CG", 1)
V_dg0 = FunctionSpace(config.domain.mesh, 'DG', 0)

# Bathymetry
bexpr = BathymetryDepthExpression('netcdf/bathymetry.nc', utm_zone=utm_zone, utm_band=utm_band)
depth = interpolate(bexpr, V_cg1) 
depth_pvd = File(os.path.join(config.params["base_path"], "bathymetry.pvd"))
depth_pvd << depth

config.params['depth'] = depth
config.turbine_function_space = V_dg0 

domains = MeshFunction("size_t", config.domain.mesh, mesh_basefile + "_physical_region.xml")
if farm_selector is not None:
  domains_ids = MeshFunction("size_t", config.domain.mesh, mesh_basefile + "_physical_region.xml")
  domains.set_all(0)
  domains.array()[domains_ids.array() == farm_selector] = 1
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
  m_opt = maximize(rf, bounds = [0, max_ct], options = {"maxiter": 600})
