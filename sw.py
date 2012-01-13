import sys
import divett
import sw_lib
from dolfin import *
from dolfin_adjoint import *

W=sw_lib.p1dgp2(divett.mesh)

state=Function(W)

state.interpolate(divett.InitialConditions())

divett.params["basename"]="p1dgp2"
#divett.params["finish_time"]=60*60*1.24 # One tidal cycle
divett.params["finish_time"]=60 # One tidal cycle
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

J = Functional(dot(state, state)*dx)
f_direct = assemble(dot(state, state)*dx)
adj_state = sw_lib.adjoint(state, divett.params, J)

ic = Function(W)
ic.interpolate(divett.InitialConditions())
def J(ic):
  state = sw_lib.timeloop_theta(M, G, rhs_contr,ufl,ufr,ic, divett.params, annotate=False)
  return assemble(dot(state, state)*dx)

minconv = test_initial_condition(J, ic, adj_state, seed=0.001)
if minconv < 1.9:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)
