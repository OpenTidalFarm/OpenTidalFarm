from dolfin import *


def rt0(mesh):
    "Return a function space U*H on mesh from the rt0 space."

    V = FunctionSpace(mesh, 'Raviart-Thomas', 1)  # Velocity space

    H = FunctionSpace(mesh, 'DG', 0)               # Height space

    return V, H


def p2p1(mesh):
    "Return a function space U*H on mesh from the P2P1 space."

    V = VectorFunctionSpace(mesh, 'CG', 2, dim=2)  # Velocity space

    H = FunctionSpace(mesh, 'CG', 1)               # Height space

    return V, H

def mini(mesh):
    "Return a function space U*H on mesh from the mini space."

    V = VectorFunctionSpace(mesh, 'CG', 1, dim=2) + VectorFunctionSpace(mesh, 'Bubble', 3) # Velocity space

    H = FunctionSpace(mesh, 'CG', 1)               # Height space

    return V, H

def p1dgp2(mesh):
    "Return a function space U*H on mesh from the P1dgP2 space."

    V = VectorFunctionSpace(mesh, 'DG', 1, dim=2)  # Velocity space

    H = FunctionSpace(mesh, 'CG', 2)               # Height space

    return V, H

def p0p1(mesh):
    "Return a function space U*H on mesh from the P0P1 space."

    V = VectorFunctionSpace(mesh, 'DG', 0, dim=2)  # Velocity space

    H = FunctionSpace(mesh, 'CG', 1)               # Height space

    return V, H

def bdfmp1dg(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDFM', 1)             # Velocity space

    H = FunctionSpace(mesh, 'DG', 1)               # Height space

    return V, H


def bdmp0(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDM', 1)              # Velocity space

    H = FunctionSpace(mesh, 'DG', 0)               # Height space

    return V, H


def bdmp1dg(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDM', 1)             # Velocity space

    H = FunctionSpace(mesh, 'DG', 1)              # Height space

    return V, H
