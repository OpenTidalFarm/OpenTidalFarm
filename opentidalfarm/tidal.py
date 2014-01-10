from dolfin import *
from dolfin_adjoint import *
from helpers import print0
# this imports NetCDFFile from netCDF4, Scientific.IO.NetCDF or scipy.io.netcdf (whichever is available)
from uptide.netcdf_reader import NetCDFFile
import utm
import uptide
import uptide.tidal_netcdf
import datetime
import scipy.interpolate

# We need to store tnci_time as a non-class variable, otherwise 
# dolfin-adjoint tries to be clever restore its values during the 
# adjoint runs which yields into unexpected behaviours in 
# the "tnci_time != self.t" statement below
tnci_time = None
class TidalForcing(Expression):
    """Create a TidalForcing Expression from OTPSnc NetCDF files, where
       the grid is stored in a separate file (with "lon_z", "lat_z" and "mz"
       fields). The actual data is read from a seperate file with hRe and hIm
       fields. """
    def __init__(self, grid_file_name, data_file_name, ranges, utm_zone, utm_band, initial_time, constituents):
        self.t = None
        self.utm_zone = utm_zone
        self.utm_band = utm_band

        tide = uptide.Tides(constituents)
        tide.set_initial_time(initial_time)
        self.tnci = uptide.tidal_netcdf.OTPSncTidalInterpolator(tide,
                    grid_file_name, data_file_name, ranges)

    def eval(self, values, X):
        global tnci_time
        if tnci_time != self.t:
            print0("Setting Tidal forcing time to %f " % self.t)
            self.tnci.set_time(self.t)
            tnci_time = self.t

        latlon = utm.to_latlon(X[0], X[1], self.utm_zone, self.utm_band)
        # OTPS has lon, lat coordinates!
        values[0] = self.tnci.get_val((latlon[1], latlon[0]), allow_extrapolation=True)


class BathymetryDepthExpression(Expression):
    """Create a bathymetry depth Expression from a lat/lon NetCDF file, where
       the depth values stored as "z" field. """
    def __init__(self, filename, utm_zone, utm_band):
        nc = NetCDFFile(filename, 'r')
        lat = nc.variables['lat']
        lon = nc.variables['lon']
        values = nc.variables['z']
        self.utm_zone = utm_zone
        self.utm_band = utm_band
        self.interpolator = scipy.interpolate.RectBivariateSpline(lat, lon, values)

    def eval(self, values, x):
        lat, lon = utm.to_latlon(x[0], x[1], self.utm_zone, self.utm_band)
        values[0] = max(10, -self.interpolator(lat, lon))
