import divett
import sw
from dolfin import *


W=sw.p1dgp2(divett.mesh)

state=Function(W)

state.interpolate(divett.InitialConditions())

divett.params["basename"]="p1dgp2"
divett.params["finish_time"]=60*60*1.24 # One tidal cycle
divett.params["dt"]=2*60
divett.params["dump_period"]=1

M,G,rhs_contr=sw.construct_shallow_water(W,divett.ds,divett.params)

state = sw.timeloop_theta(M,G,rhs_contr,state,divett.params)

