import sys
import sw_config 
import sw_lib
from dolfin import *
from dolfin_adjoint import *

config = sw_config.SWConfiguration(nx=40, ny=3) 
config.params["basename"]="p1dgp2"
config.params["finish_time"]=2*pi/(sqrt(config.params["g"]*config.params["depth"])*config.params["k"])-0.000000000001
config.params["dt"]=config.params["finish_time"]/100
config.params["period"]=60*60*1.24
config.params["dump_period"]=1

class InitialConditions(Expression):
    def __init__(self):
        pass
    def eval(self, values, X):
        values[0]=config.params['eta0']*sqrt(config.params['g']*config.params['depth'])*cos(config.params["k"]*X[0])
        values[1]=0.
        values[2]=config.params['eta0']*cos(config.params["k"]*X[0])
    def value_shape(self):
        return (3,)

config.InitialConditions = InitialConditions

def error(config):
  W=sw_lib.p1dgp2(config.mesh)
  initstate=Function(W)
  initstate.interpolate(config.InitialConditions())

  M,G,rhs_contr,ufl,ufr=sw_lib.construct_shallow_water(W, config.ds, config.params)
  finalstate = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, ufr, initstate, config.params, annotate=False)

  analytic_sol = Expression(("eta0*sqrt(g*depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", \
                             "0", \
                             "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"), \
                             eta0=config.params["eta0"], g=config.params["g"], \
                             depth=config.params["depth"], t=config.params["finish_time"], k=config.params["k"])
  exactstate=Function(W)
  exactstate.interpolate(analytic_sol)
  e = finalstate-exactstate
  return sqrt(assemble(dot(e,e)*dx))

print "Error ", error(config)
sys.exit()

adj_html("sw_forward.html", "forward")
adj_html("sw_adjoint.html", "adjoint")
sw_lib.replay(state, config.params)

J = Functional(dot(state, state)*dx)
f_direct = assemble(dot(state, state)*dx)
adj_state = sw_lib.adjoint(state, config.params, J)

ic = Function(W)
ic.interpolate(config.InitialConditions())
def J(ic):
  state = sw_lib.timeloop_theta(M, G, rhs_contr,ufl,ufr,ic, config.params, annotate=False)
  analytic_sol = Expression("eta0*cos(pi/3000*x[0]-sqrt(g/h)*pi/3000*t")
  return assemble(dot(state, state)*dx)

minconv = test_initial_condition(J, ic, adj_state, seed=0.001)
if minconv < 1.9:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)
