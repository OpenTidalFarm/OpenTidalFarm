import sys
import divett
import sw_lib
from dolfin import *
from dolfin_adjoint import *

#W=sw_lib.p1dgp2(divett.generate_mesh)

#state=Function(W)

#state.interpolate(divett.InitialConditions())

divett.params["basename"]="p1dgp2"
divett.params["finish_time"]=2*pi/(sqrt(divett.params["g"]*divett.params["depth"])*pi/3000)
divett.params["dt"]=divett.params["finish_time"]/100
divett.params["period"]=60*60*1.24
divett.params["dump_period"]=1

#M,G,rhs_contr,ufl,ufr=sw_lib.construct_shallow_water(W,divett.ds,divett.params)

#state = sw_lib.timeloop_theta(M,G,rhs_contr,ufl,ufr,state,divett.params)

def error(mesh):
  W=sw_lib.p1dgp2(mesh)
  initstate=Function(W)
  initstate.interpolate(divett.InitialConditions())

  M,G,rhs_contr,ufl,ufr=sw_lib.construct_shallow_water(W,divett.ds,divett.params)
  finalstate = sw_lib.timeloop_theta(M, G, rhs_contr,ufl,ufr,state, divett.params, annotate=False)
  analytic_sol = Expression(("eta0*cos(pi/3000*x[0]-sqrt(g/depth)*pi/3000*t)", \
                             "0", \
                             "eta0*sqrt(g/depth)*cos(pi/3000*x[0]-sqrt(g/depth)*pi/3000*t)"), \
                             eta0=divett.params["eta0"], g=divett.params["g"], \
                             depth=divett.params["depth"], t=divett.params["finish_time"])
  exctstate=Function(W)
  exctstate.interpolate(analytic_sol)
  e = finalstate-exctstate
  x = inner(e,e)*dx
  return sqrt(assemble(dot(e,e)*dx))

print "Error ", error(divett.get_mesh(20, 3))
sys.exit()

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
  from IPython.Shell import IPShellEmbed
  ipshell = IPShellEmbed()
  ipshell()
  analytic_sol = Expression("eta0*cos(pi/3000*x[0]-sqrt(g/h)*pi/3000*t")
  return assemble(dot(state, state)*dx)

minconv = test_initial_condition(J, ic, adj_state, seed=0.001)
if minconv < 1.9:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)
