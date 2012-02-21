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
    '''Solve (1+turbine)*M*state = (2, 0, 0)*M. 
       The solutioin is a x-velocity of 2/(turbine+1) and a pressure and y-velocity of 0. 
    '''
    
    u_out, p_out = sw_lib.output_files(params["basename"])
    M_u_out, v_out, u_out_state=sw_lib.u_output_projector(state.function_space())
    M_p_out, q_out, p_out_state=sw_lib.p_output_projector(state.function_space())

    tmpstate=Function(state.function_space())

    rhs = action(M, state)
    # Solve the mini model 
    solver_parameters = {"linear_solver": "cg", "preconditioner": "sor"}
    solve(A==rhs, tmpstate, solver_parameters=solver_parameters, annotate=annotate)

    state.assign(tmpstate, annotate=annotate)

    # Project the solution to P1 for visualisation.
    rhs_p = assemble(inner(v_out,state.split()[0])*dx)
    solve(M_u_out, u_out_state.vector(), rhs_p, "cg", "sor", annotate=False) 

    # Project the solution to P1 for visualisation.
    rhs_p = assemble(inner(q_out,state.split()[1])*dx)
    solve(M_p_out, p_out_state.vector(), rhs_p, "cg", "sor", annotate=False) 
    
    u_out << u_out_state
    p_out << p_out_state

    if time_functional is not None:
      j = assemble(time_functional.Jt(state)) 
      djdm = numpy.array([assemble(f) for f in time_functional.dJtdm(state)])
      return j, djdm, state 
    else: 
      return state

