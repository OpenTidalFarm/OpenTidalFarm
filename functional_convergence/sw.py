'''Tests the convergence of the power output with increasing resolution.
   It also checks that the power output is independent of minimal movement of its position,
   to ensure that no numerical effects influence its value.'''
import sys
import sw_config 
import sw_lib
from turbines import *
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint

set_log_level(30)
myid = MPI.process_number()

def run_model(nx, ny, turbine_model, turbine_pos):
  config = sw_config.SWConfiguration(nx, ny)
  period = 1.24*60*60 # Wave period
  config.params["k"]=2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  config.params["basename"]="p1dgp2"
  config.params["finish_time"]=2./4*period
  config.params["dt"]=config.params["finish_time"]/40
  config.params["dump_period"]=1
  config.params["bctype"]="flather"

  # Start at rest state
  config.params["start_time"] = period/4 

  # Turbine settings
  config.params["friction"]=0.0025
  config.params["turbine_pos"] = [[1000., 500.], [2000., 500.]]
  config.params["turbine_friction"] = 12./config.params["depth"]
  config.params["turbine_length"] = 200
  config.params["turbine_width"] = 200

  # Now create the turbine measure
  config.initialise_turbines_measure()

  ############# Initial Conditions ##################
  class InitialConditions(Expression):
      def __init__(self):
          pass
      def eval(self, values, X):
          values[0]=config.params['eta0']*sqrt(config.params['g']*config.params['depth'])*cos(config.params["k"]*X[0]-sqrt(config.params["g"]*config.params["depth"])*config.params["k"]*config.params["start_time"])
          values[1]=0.
          values[2]=config.params['eta0']*cos(config.params["k"]*X[0]-sqrt(config.params["g"]*config.params["depth"])*config.params["k"]*config.params["start_time"])
      def value_shape(self):
          return (3,)

  W=sw_lib.p1dgp2(config.mesh)

  state=Function(W)
  state.interpolate(InitialConditions())

  tf = Function(W)
  tf.interpolate(turbine_model(config))
  sw_lib.save_to_file(tf, "turbines")

  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf[0])
  def functional(state):
    return config.params["dt"]*0.5*config.params["turbine_friction"]*(dot(state[0], state[0]) + dot(state[1], state[1])/(config.params["g"]*config.params["depth"]))**1.5*config.dx(1)

  j, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)
  return j

def refine(nx, ny, level=0.3):
  return int(float(nx)/level), int(float(ny)/level)

myid = MPI.process_number()
turbine_pos = [[1000., 500.], [2000., 500.]]
nx_orig = 60
ny_orig = 20

print "Turbine size: 200x200"

for name, model in {"RectangleTurbine": RectangleTurbines, "GaussianTurbine": GaussianTurbines}.iteritems():
  nx, ny = (nx_orig, ny_orig)
  j = run_model(nx_orig, ny_orig, model, turbine_pos)
  if myid == 0:
    print name 
    print "%i x %i \t\t| %.4g " % (nx, ny, j)

  nx, ny = refine(nx_orig, ny_orig)
  j = run_model(nx, ny, model, turbine_pos)
  if myid == 0:
    print "%i x %ii \t\t| %.4g " % (nx, ny, j)

  nx, ny = refine(nx, ny)
  j = run_model(nx, ny, model, turbine_pos)
  if myid == 0:
    print "%i x %i \t\t| %.4g " % (nx, ny, j)

