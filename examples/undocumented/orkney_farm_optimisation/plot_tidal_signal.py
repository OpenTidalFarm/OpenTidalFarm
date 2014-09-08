import uptide
import uptide.tidal
import uptide.tidal_netcdf
import datetime
from matplotlib.pyplot import plot, show
from numpy import arange

constituents = ['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2']
tide = uptide.Tides(constituents)
tide.set_initial_time(datetime.datetime(2009,4,1,0,0,0))
tnci = uptide.tidal_netcdf.OTPSncTidalInterpolator(tide,
                                                   'gridES2008.nc', 'hf.ES2008.nc', ranges=((-4.0,0.0),(58.0,61.0)))

trange = arange(0,86400*30,600)
etas=[]

for t in trange:
  tnci.set_time(t)
  etas.append(tnci.get_val((-3.701170833257142, 58.61448248825699), allow_extrapolation=True))

plot(trange/86400., etas)
show()
