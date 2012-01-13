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

    uc = 2.0
    etac = sqrt(params["depth"]/params["g"])*uc

    # The dirichlet boundary condition on the left hand side 
    ufl = Expression("2*sin(2*pi*t/period)", t=params["current_time"], period=params["period"])
    rhs_contr=inner(ufl*n,q*n)*ds(1)

    # The contributions of the Flather boundary condition on the right hand side
    ufr = Expression("uc-sqrt(g/depth)*etac", uc=uc, etac=etac, t=params["current_time"], g=params["g"], depth=params["depth"])
    rhs_contr+=inner(ufr*n,q*n)*ds(2)
    Ct+=inner(-sqrt(params["g"]/params["depth"])*h,q)*ds(2)

    # Pressure gradient operator
    C=(params["g"]*params["depth"])*\
        inner(v,grad(h))*dx+inner(avg(v),jump(h,n))*dS
    C+=inner(v,h*n)*ds(1)
    C+=inner(v,h*n)*ds(2)

    # Add the bottom friction
    class FrictionExpr(Expression):
        def eval(self, value, x):
            friction = params["friction"] 

            # Check if x lies in a position where a turbine is deployed and if, then increase the friction
            x_pos = numpy.array(params["turbine_pos"])[:,0] 
            x_pos_low = x_pos-params["turbine_length"]/2
            x_pos_high = x_pos+params["turbine_length"]/2

            y_pos = numpy.array(params["turbine_pos"])[:,1] 
            y_pos_low = y_pos-params["turbine_length"]/2
            y_pos_high = y_pos+params["turbine_length"]/2
            if ((x_pos_low < x[0]) & (x_pos_high > x[0]) & (y_pos_low < x[1]) & (y_pos_high > x[1])).any():
              friction += params["turbine_friction"] 

            value[0] = friction 

    friction = FrictionExpr()

    R=friction*inner(2*u,v)*dx # TODO: Replace 2 by |u|

    try:
        # Coriolis term
        F=params["f"]*inner(v,as_vector([-u[1],u[0]]))*dx      
    except KeyError:
        F=0

    if params.has_key("big_spring"):
        print "big spring active: ", params["big_spring"]
        C+=inner(v,n)*inner(u,n)*params["big_spring"]*ds

    return (M, C+Ct+F+R, rhs_contr, ufl, ufr)

def timeloop_theta(M, G, rhs_contr, ufl, ufr, state, params):
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
        ufl.t=t # Update time for the Boundary condition expression
        ufr.t=t # Update time for the Boundary condition expression
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
            

