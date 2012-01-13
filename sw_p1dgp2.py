import divett
import sw
from dolfin import *


W=sw.p1dgp2(divett.mesh)

state=Function(W)

state.interpolate(divett.InitialConditions())

divett.params["basename"]="p1dgp2"
divett.params["finish_time"]=60*60*1.24 # One tidal cycle
divett.params["dt"]=30
divett.params["period"]=60*60*1.24
divett.params["dump_period"]=1
divett.params["friction"]=0.0025
divett.params["turbine_pos"]=[[200., 500.], [1000., 700.]]
divett.params["turbine_length"] = 20.
divett.params["turbine_depth"] = 5.
divett.params["turbine_friction"] = 12.

M,G,rhs_contr,ufl,ufr=sw.construct_shallow_water(W,divett.ds,divett.params)

state = sw.timeloop_theta(M,G,rhs_contr,ufl,ufr,state,divett.params)

