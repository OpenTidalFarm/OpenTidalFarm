from dolfin import *

def rt0(mesh):
    "Return a function space U*H on mesh from the rt0 space."

    V = FunctionSpace(mesh, 'Raviart-Thomas', 1) # Velocity space
 
    H = FunctionSpace(mesh, 'DG', 0)             # Height space

    W=V*H                                        # Mixed space of both.

    return W

def p2p1(mesh):
    "Return a function space U*H on mesh from the p2p1 space."

    V = VectorFunctionSpace(mesh, 'CG', 2, dim=2)# Velocity space
 
    H = FunctionSpace(mesh, 'CG', 1)             # Height space

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

    W = V*H                                        # Mixed space of both.

    return W
