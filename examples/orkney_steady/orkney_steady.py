from opentidalfarm import *
import ipopt
import utm
import uptide
import uptide.tidal_netcdf
from uptide.netcdf_reader import NetCDFInterpolator
import datetime

utm_zone = 30
utm_band = 'V'

import sys
set_log_level(PROGRESS)

config = SteadyConfiguration("mesh/coast_idBoundary_utm.xml", [1, 1])
config.params['diffusion_coef'] = 180.0
config.params['save_checkpoints'] = True

# Tidal boundary forcing
bc = DirichletBCSet(config)
class TidalForcing(Expression):
    def __init__(self):
        self.t = None
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
bc.add_constant_eta(1, eta_expr)
bc.add_constant_eta(2, eta_expr)
# comment out if you want free slip:
#bc.add_noslip_u(3)
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


# Place some turbines 
#deploy_turbines(config, nx = 32, ny = 8)
#config.params["turbine_friction"] = 0.5*numpy.array(config.params["turbine_friction"]) 

config.info()
import sys; sys.exit(1)
rf = ReducedFunctional(config)

# Load checkpoints if desired by the user
if len(sys.argv) > 1 and sys.argv[1] == "--from-checkpoint":
  rf.load_checkpoint("checkpoint")

# Get the upper and lower bounds for the turbine positions
lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config)

parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP", options = {"maxiter": 300, "ftol": 1.0}) 
