import divett
import sw_lib
from dolfin import *
from dolfin_adjoint import *

W=sw_lib.p1dgp2(divett.mesh)

state=Function(W)

state.interpolate(divett.InitialConditions())

divett.params["basename"]="p1dgp2"
#divett.params["finish_time"]=60*60*1.24 # One tidal cycle
divett.params["finish_time"]=2*60*1.24 # One tidal cycle
divett.params["dt"]=30
divett.params["period"]=60*60*1.24
divett.params["dump_period"]=1
divett.params["friction"]=0.0025
divett.params["turbine_pos"]=[[200., 500.], [1000., 700.]]
divett.params["turbine_length"] = 20.
divett.params["turbine_depth"] = 5.
divett.params["turbine_friction"] = 12.

M,G,rhs_contr,ufl,ufr=sw_lib.construct_shallow_water(W,divett.ds,divett.params)

state = sw_lib.timeloop_theta(M,G,rhs_contr,ufl,ufr,state,divett.params)


adj_html("sw_forward.html", "forward")
adj_html("sw_adjoint.html", "adjoint")

sw_lib.replay(state, divett.params)
