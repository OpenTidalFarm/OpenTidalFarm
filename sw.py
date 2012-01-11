from dolfin import *
import numpy
import sys

class parameters(dict):
    '''Parameter dictionary. This subclasses dict so defaults can be set.'''
    def __init__(self, dict={}):
        self["current_time"]=0.0
        self["theta"]=0.5

        # Apply dict after defaults so as to overwrite the defaults
        for key,val in dict.iteritems():
            self[key]=val

        self.required={
            "depth" : "water depth",
            "dt" : "timestep",
            "finish_time" : "finish time",
            "dump_period" : "dump period in timesteps",
            "basename" : "base name for I/O"
            }

    def check(self):
        for key, error in self.required.iteritems():
            if not self.has_key(key):
                sys.stderr.write("Missing parameter: "+key+"\n"+
                                 "This is used to set the "+error+"\n")
                raise KeyError

def rt0(mesh):
    "Return a function space U*H on mesh from the rt0 space."

    V = FunctionSpace(mesh, 'Raviart-Thomas', 1) # Velocity space
 
    H = FunctionSpace(mesh, 'DG', 0)             # Height space

    W=V*H                                        # Mixed space of both.

    return W
def p1dgp2(mesh):
    "Return a function space U*H on mesh from the rt0 space."

    V = VectorFunctionSpace(mesh, 'DG', 1, dim=2)# Velocity space
 
    H = FunctionSpace(mesh, 'CG', 2)             # Height space

    W=V*H                                        # Mixed space of both.

    return W

def bdfmp1dg(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDFM', 1)# Velocity space
 
    H = FunctionSpace(mesh, 'DG', 1)             # Height space

    W=V*H                                        # Mixed space of both.

    return W

def bdmp0(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDM', 1)# Velocity space
 
    H = FunctionSpace(mesh, 'DG', 0)             # Height space

    W=V*H                                        # Mixed space of both.

    return W

def bdmp1dg(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDM', 1)# Velocity space
 
    H = FunctionSpace(mesh, 'DG', 1)             # Height space

    W=V*H                                        # Mixed space of both.

    return W

def construct_shallow_water(W,ds,params):
    """Construct the linear shallow water equations for the space W(=U*H) and a
    dictionary of parameters params."""
    # Sanity check for parameters.
    params.check()

    (v, q)=TestFunctions(W)
    (u, h)=TrialFunctions(W)

    n=FacetNormal(W.mesh())

    # Mass matrix
    M=inner(v,u)*dx
    M+=inner(q,h)*dx

    # Divergence term.
    Ct=-inner(u,grad(q))*dx+inner(avg(u),jump(q,n))*dS
    # Add a weak dirichlet boundary condition 
    #Ct+=inner(u/2,q*n)*ds(1)
    Ct+=inner(u,q*n)*ds(2)
    uf = Expression("sqrt(g/depth)*cos(t)", t=0., g=params["g"], depth=params["depth"])
    rhs_contr=inner(uf*n,q*n)*ds(1)
    #rhs_contr=inner(Constant("-1.")*n,q*n)*ds(2)

    #Ct=div(u)*q*dx
    # Pressure gradient operator
    C=(params["g"]*params["depth"])*\
        inner(v,grad(h))*dx+inner(avg(v),jump(h,n))*dS
    #C+=inner(v,h*n)*ds(1)
    #C+=inner(v,h*n)*ds(2)
    #rhs_contr=inner(v,Constant("-0.5")*n)*ds(1)

    try:
        # Coriolis term
        F=params["f"]*inner(v,as_vector([-u[1],u[0]]))*dx      
    except KeyError:
        F=0

    if params.has_key("big_spring"):
        print "big spring active: ", params["big_spring"]
        C+=inner(v,n)*inner(u,n)*params["big_spring"]*ds

    return (M, C+Ct+F, rhs_contr, uf)

def timeloop_theta(M, G, rhs_contr, uf, state, params):
    '''Solve M*dstate/dt = G*state using a theta scheme.'''
    
    A=M+params["theta"]*params["dt"]*G

    A_r=M-(1-params["theta"])*params["dt"]*G

    u_out,p_out=output_files(params["basename"])

    M_u_out, v_out, u_out_state=u_output_projector(state.function_space())

    M_p_out, q_out, p_out_state=p_output_projector(state.function_space())

    # Project the solution to P1 for visualisation.
    rhs=assemble(inner(v_out,state.split()[0])*dx)
    solve(M_u_out, u_out_state.vector(),rhs,"cg","sor") 
    
    # Project the solution to P1 for visualisation.
    rhs=assemble(inner(q_out,state.split()[1])*dx)
    solve(M_p_out, p_out_state.vector(),rhs,"cg","sor") 
    
    u_out << u_out_state
    p_out << p_out_state
    
    t = params["current_time"]
    dt= params["dt"]
    
    step=0    

    tmpstate=Function(state.function_space())

    while (t < params["finish_time"]):
        t+=dt
        uf.t=t # Update time for the Boundary condition expression
        step+=1
        rhs=action(A_r,state)+params["dt"]*rhs_contr
        
        # Solve the shallow water equations.
        solve(A==rhs, tmpstate)
        #solve(A, state.vector(), rhs, "preonly", "lu")

        state.assign(tmpstate)

        if step%params["dump_period"] == 0:
        
            # Project the solution to P1 for visualisation.
            rhs=assemble(inner(v_out,state.split()[0])*dx)
            solve(M_u_out, u_out_state.vector(),rhs,"cg","sor") 

            # Project the solution to P1 for visualisation.
            rhs=assemble(inner(q_out,state.split()[1])*dx)
            solve(M_p_out, p_out_state.vector(),rhs,"cg","sor") 
            
            u_out << u_out_state
            p_out << p_out_state

    return state # return the state at the final time

def u_output_projector(W):
    # Projection operator for output.
    Output_V=VectorFunctionSpace(W.mesh(), 'CG', 1, dim=2)
    
    u_out=TrialFunction(Output_V)
    v_out=TestFunction(Output_V)
    
    M_out=assemble(inner(v_out,u_out)*dx)
    
    out_state=Function(Output_V)

    return M_out, v_out, out_state

def p_output_projector(W):
    # Projection operator for output.
    Output_V=FunctionSpace(W.mesh(), 'CG', 1)
    
    u_out=TrialFunction(Output_V)
    v_out=TestFunction(Output_V)
    
    M_out=assemble(inner(v_out,u_out)*dx)
    
    out_state=Function(Output_V)

    return M_out, v_out, out_state

def output_files(basename):
        
    # Output file
    u_out = File(basename+"_u.pvd", "compressed")
    p_out = File(basename+"_p.pvd", "compressed")

    return u_out, p_out
            

