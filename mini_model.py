import sw_lib
import numpy
from dolfin import *
from dolfin_adjoint import *

def construct_mini_model(W, params, turbine_field):
    (v, q)=TestFunctions(W)
    (u, h)=TrialFunctions(W)

    # Mass matrix
    M=inner(v,u)*dx
    M+=inner(q,h)*dx

    A = (1.0+turbine_field)*inner(v,u)*dx
    A += inner(q,h)*dx

    return A, M

def mini_model(A, M, state, params, time_functional=None, annotate=True):
    '''Solve (1+turbine)*M*state = M*old_state. 
       The solution is a x-velocity of old_state/(turbine_friction + 1) and a zero pressure value y-velocity.
    '''
    
    u_out, p_out = sw_lib.output_files(params["element_type"].func_name)
    M_u_out, v_out, u_out_state=sw_lib.u_output_projector(state.function_space())
    M_p_out, q_out, p_out_state=sw_lib.p_output_projector(state.function_space())

    if time_functional is not None:
      fac = 0.0
      j = fac*0.5*params["dt"]*assemble(time_functional.Jt(state)) 
      djdm = fac*0.5*params["dt"]*numpy.array([assemble(f) for f in time_functional.dJtdm(state)])

    tmpstate=Function(state.function_space(), name="tmp_state")
    rhs = action(M, state)
    # Solve the mini model 
    solver_parameters = {"linear_solver": "cg", "preconditioner": "sor"}
    solve(A==rhs, tmpstate, solver_parameters=solver_parameters, annotate = annotate)

    state.assign(tmpstate, annotate=annotate)

    # Project the solution to P1 for visualisation.
    rhs_p = assemble(inner(v_out,state.split()[0])*dx)
    solve(M_u_out, u_out_state.vector(), rhs_p, "cg", "sor", annotate=False) 

    # Project the solution to P1 for visualisation.
    rhs_p = assemble(inner(q_out,state.split()[1])*dx)
    solve(M_p_out, p_out_state.vector(), rhs_p, "cg", "sor", annotate=False) 
    
    u_out << u_out_state
    p_out << p_out_state

    # Bump timestep to shut up libadjoint.
    adj_inc_timestep()

    if time_functional is not None:
      j += 0.5*params["dt"]*assemble(time_functional.Jt(state)) 
      djdm += 0.5*params["dt"]*numpy.array([assemble(f) for f in time_functional.dJtdm(state)])
      print j, djdm
      return j, djdm

def mini_model_solve(W, config, state, turbine_field=None, time_functional=None, annotate=True, linear_solver="default", preconditioner="default", u_source = None):
    A, M = construct_mini_model(W, config.params, turbine_field)
    return mini_model(A, M, state, params = config.params, time_functional = time_functional)
