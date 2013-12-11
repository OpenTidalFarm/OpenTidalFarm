from dolfin import *
from dolfin_adjoint import *
from opentidalfarm import print0
from uptide.netcdf_reader import NetCDFInterpolator
import utm
import uptide
import uptide.tidal_netcdf
import datetime

utm_zone = 30
utm_band = 'V'

# We need to store tnci_time as a non-class variable, otherwise 
# dolfin-adjoint tries to be clever restore its values during the 
# adjoint runs which yields into unexpected behaviours in 
# the "tnci_time != self.t" statement below
tnci_time = None
class TidalForcing(Expression):
    def __init__(self):
        self.t = None

        constituents = ['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2']
        tide = uptide.Tides(constituents)
        tide.set_initial_time(datetime.datetime(2001,9,18,0,0,0))
        self.tnci = uptide.tidal_netcdf.OTPSncTidalInterpolator(tide,
                    'gridES2008.nc', 'hf.ES2008.nc', ranges=((-4.0,0.0),(58.0,61.0)))

    def eval(self, values, X):
        global tnci_time
        if tnci_time != self.t:
            print0("Setting Tidal forcing time to %f " % self.t)
            self.tnci.set_time(self.t)
            tnci_time = self.t

        latlon = utm.to_latlon(X[0], X[1], utm_zone, utm_band)
        # OTPS has lon, lat coordinates!
        values[0] = self.tnci.get_val((latlon[1], latlon[0]), allow_extrapolation=True)


class BathymetryDepthExpression(Expression):
  def __init__(self, filename):
    self.nci = NetCDFInterpolator(filename, ('lat', 'lon'), ('lat', 'lon'))
    self.nci.set_field("z")

  def eval(self, values, x):
    latlon = utm.to_latlon(x[0], x[1], utm_zone, utm_band)
    values[0] = max(10, -self.nci.get_val(latlon))

