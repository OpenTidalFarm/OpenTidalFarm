import sys
import divett
import sw_lib
from dolfin import *
from dolfin_adjoint import *

W=sw_lib.p1dgp2(divett.mesh)

state=Function(W)


period = 1.24*60*60 # Wave period
divett.params["k"]=2*pi/(period*sqrt(divett.params["g"]*divett.params["depth"]))
divett.params["basename"]="p1dgp2"
divett.params["finish_time"]=period
divett.params["dt"]=divett.params["finish_time"]/100
print "Wave period (in h): ", period/60/60 
divett.params["dump_period"]=1

class InitialConditions(Expression):
    def __init__(self):
        pass
    def eval(self, values, X):
        values[0]=divett.params['eta0']*sqrt(divett.params['g']*divett.params['depth'])*cos(divett.params["k"]*X[0])
        values[1]=0.
        values[2]=divett.params['eta0']*cos(divett.params["k"]*X[0])
    def value_shape(self):
        return (3,)

state.interpolate(InitialConditions())

M,G,rhs_contr,ufl,ufr=sw_lib.construct_shallow_water(W,divett.ds,divett.params)

state = sw_lib.timeloop_theta(M,G,rhs_contr,ufl,ufr,state,divett.params)
sys.exit()

adj_html("sw_forward.html", "forward")
adj_html("sw_adjoint.html", "adjoint")
sw_lib.replay(state, divett.params)

J = Functional(dot(state, state)*dx)
f_direct = assemble(dot(state, state)*dx)
adj_state = sw_lib.adjoint(state, divett.params, J)

ic = Function(W)
ic.interpolate(InitialConditions())
def J(ic):
  state = sw_lib.timeloop_theta(M, G, rhs_contr,ufl,ufr,ic, divett.params, annotate=False)
  return assemble(dot(state, state)*dx)

minconv = test_initial_condition(J, ic, adj_state, seed=0.001)
if minconv < 1.9:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)
