import shallow_water_model as sw_model
import numpy
import helpers
from dolfin import *
from dolfin_adjoint import *

def construct_mini_model(config, turbine_field):
    (v, q) = TestFunctions(config.function_space)
    (u, h) = TrialFunctions(config.function_space)

    # Mass matrix
    M = inner(v,u)*dx
    M += inner(q,h)*dx

    A = (1.0+turbine_field)*inner(v,u)*dx
    A += inner(q,h)*dx

    return A, M

def mini_model(A, M, state, params, functional=None, annotate=True):
    '''Solve (1+turbine)*M*state = M*old_state. 
       The solution is a x-velocity of old_state/(turbine_friction + 1) and a zero pressure value y-velocity.
    '''
    
    if functional is not None and not params["functional_final_time_only"]:
        j = 0.5*params["dt"]*assemble(functional.Jt(state)) 

    adjointer.time.start(0.0)
    tmpstate = Function(state.function_space(), name="tmp_state")
    rhs = action(M, state)
    # Solve the mini model 
    solver_parameters = {"linear_solver": "cg", "preconditioner": "sor"}
    solve(A == rhs, tmpstate, solver_parameters=solver_parameters, annotate = annotate)

    state.assign(tmpstate, annotate=annotate)

    # Bump timestep to shut up libadjoint.
    adj_inc_timestep(time=params["dt"], finished = True)

    if functional is not None:
      if params["functional_final_time_only"]:
        j = assemble(functional.Jt(state)) 
      else:
        j += 0.5*params["dt"]*assemble(functional.Jt(state)) 
      return j

def mini_model_solve(config, state, turbine_field=None, functional=None, annotate=True, linear_solver="default", preconditioner="default", u_source = None):
    A, M = construct_mini_model(config, turbine_field)
    return mini_model(A, M, state, params = config.params, functional = functional)
