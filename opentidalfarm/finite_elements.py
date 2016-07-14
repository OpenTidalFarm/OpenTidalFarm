from dolfin import *


def rt0():
    "Return a function space U*H on mesh from the rt0 space."

    V = FiniteElement('Raviart-Thomas', triangle, 1)  # Velocity space

    H = FiniteElement('DG', triangle, 0)               # Height space

    return V, H


def p2p1():
    "Return a function space U*H on mesh from the P2P1 space."

    V = VectorElement('CG', triangle, 2, dim=2)  # Velocity space

    H = FiniteElement('CG', triangle, 1)               # Height space

    return V, H

def mini():
    "Return a function space U*H on mesh from the mini space."

    V = VectorElement('CG', triangle, 1, dim=2) + VectorElement('Bubble', triangle, 3) # Velocity space

    H = FiniteElement('CG', triangle, 1)               # Height space

    return V, H

def p1dgp2():
    "Return a function space U*H on mesh from the P1dgP2 space."

    V = VectorElement('DG', triangle, 1, dim=2)  # Velocity space

    H = FiniteElement('CG', triangle, 2)               # Height space

    return V, H

def p0p1():
    "Return a function space U*H on mesh from the P0P1 space."

    V = VectorElement('DG', triangle, 0, dim=2)  # Velocity space

    H = FiniteElement('CG', triangle, 1)               # Height space

    return V, H

def bdfmp1dg():
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDFM', 1)             # Velocity space

    H = FiniteElement('DG', triangle, 1)               # Height space

    return V, H


def bdmp0():
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDM', 1)              # Velocity space

    H = FiniteElement('DG', triangle, 0)               # Height space

    return V, H


def bdmp1dg():
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDM', 1)             # Velocity space

    H = FiniteElement('DG', triangle, 1)              # Height space

    return V, H
