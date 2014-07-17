from opentidalfarm import *
import datetime
from math import pi
import os.path
import sys
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
config.params['viscosity'] = 180.0
config.params["controls"] = ["turbine_friction"]
config.params["turbine_parametrisation"] = "smeared"
config.params["automatic_scaling"] = False 
config.params['friction'] = Constant(0.0025)
config.params['base_path'] = "forward_only"

config.params['start_time'] = 0 
config.params['dt'] = 600 
config.params['finish_time'] = 12.5 * 60 * 60 
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

config.info()

rf = ReducedFunctional(config, scale=-1e-6)

print "Running forward model"
m0 = rf.initial_control()
rf.j(m0, annotate=False)
print "Finished"
