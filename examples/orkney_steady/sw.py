from opentidalfarm import *
import utm
import uptide
import uptide.tidal_netcdf
from uptide.netcdf_reader import NetCDFInterpolator
import datetime
from pyOpt import SLSQP

utm_zone = 30
utm_band = 'V'

import sys
set_log_level(PROGRESS)

config = SteadyConfiguration("mesh/coast_idBoundary_utm.xml", [1, 1])
config.params['viscosity'] = Constant(360.0)
config.params['save_checkpoints'] = True
config.params['turbine_x'] = 100
config.params['turbine_y'] = 100

# Tidal boundary forcing
bc = DirichletBCSet(config)
class TidalForcing(Expression):
    def __init__(self):
        self.t = 0
        self.tnci_time = None

        constituents = ['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2']
        tide = uptide.Tides(constituents)
        tide.set_initial_time(datetime.datetime(2001,9,18,0,0,0))
        self.tnci = uptide.tidal_netcdf.OTPSncTidalInterpolator(tide,
                    'gridES2008.nc', 'hf.ES2008.nc', ranges=((-4.0,0.0),(58.0,61.0)))

    def eval(self, values, X):
        if self.tnci_time != self.t:    
            self.tnci.set_time(self.t)
            self.tnci_time = self.t

        latlon = utm.to_latlon(X[0], X[1], utm_zone, utm_band)
        # OTPS has lon, lat coordinates!
        values[0] = self.tnci.get_val((latlon[1], latlon[0]), allow_extrapolation=True)

eta_expr = TidalForcing() 
bc.add_analytic_eta(1, eta_expr) # inflow
bc.add_analytic_eta(2, eta_expr) # outflow
# comment if you want free slip:
bc.add_noslip_u(3)               # coast
config.params['bctype'] = 'strong_dirichlet'
config.params['strong_bc'] = bc

# Bathymetry
class BathymetryDepthExpression(Expression):
  def __init__(self, filename):
    self.nci = NetCDFInterpolator(filename, ('lat', 'lon'), ('lat', 'lon'))
    self.nci.set_field("z")

  def eval(self, values, x):
    latlon = utm.to_latlon(x[0], x[1], utm_zone, utm_band)
    values[0] = max(10, -self.nci.get_val(latlon))

bexpr = BathymetryDepthExpression('bathymetry/bathymetry.nc')
depth = interpolate(bexpr, FunctionSpace(config.domain.mesh, "CG", 1))
depth_pvd = File("bathymetry.pvd")
depth_pvd << depth

config.params['depth'] = depth
domains = MeshFunction("size_t", config.domain.mesh, "mesh/coast_idBoundary_utm_physical_region.xml")
config.site_dx = Measure("dx")[domains]

site_x_start = 491420
site_y_end = 6.50241e6
site_x_end = 493735
site_y_start = 6.50201e6

feasible_area = get_distance_function(config, domains)
File("feasible_area.pvd") << feasible_area
feasible_constraint = get_domain_constraints(config, feasible_area, attraction_center=(0.5*(site_x_start + site_x_end), 0.5*(site_y_start + site_y_end)))
distance_constraint = get_minimum_distance_constraint_func(config)

constraints = [feasible_constraint, distance_constraint]

# Place some turbines 
config.set_site_dimensions(site_x_start, site_x_end, site_y_start, site_y_end)
deploy_turbines(config, nx = 8, ny = 2, friction=10.5)

config.info()
rf = ReducedFunctional(config)

#m0 = rf.initial_control()
#rf.update_turbine_cache(m0)
#File("turbines.pvd") << config.turbine_cache.cache["turbine_field"]
#import sys; sys.exit()

# Load checkpoints if desired by the user
if len(sys.argv) > 1 and sys.argv[1] == "--from-checkpoint":
  rf.load_checkpoint("checkpoint")

parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'

nlp, grad = rf.pyopt_problem(constraints=constraints)
slsqp = SLSQP(options={"MAXIT": 300})
res = slsqp(nlp, sens_type=grad)
