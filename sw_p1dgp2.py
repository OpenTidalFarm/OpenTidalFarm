import divett
import sw
from dolfin import *


W=sw.p1dgp2(divett.mesh)

state=Function(W)

state.interpolate(divett.InitialConditions())

divett.params["basename"]="p1dgp2"
divett.params["finish_time"]=divett.params["dt"]*10
divett.params["finish_time"]=divett.params["dt"]*2
divett.params["dump_period"]=1

M,G=sw.construct_shallow_water(W,divett.params)

state = sw.timeloop_theta(M,G,state,divett.params)

