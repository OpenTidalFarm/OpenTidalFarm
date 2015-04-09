from opentidalfarm import *

class VorticitySolver(object):
    def __init__(self, V):
        self.u = Function(V)
        Q = V.extract_sub_space([0]).collapse()

        r = TrialFunction(Q)
        s = TestFunction(Q)
        a = r*s*dx
        self.L = (self.u[0].dx(1) - self.u[1].dx(0))*s*dx
        self.a_mat = assemble(a)

        self.vort = Function(Q)

    def solve(self, u, annotate=False):
        self.u.assign(u)
        L_mat = assemble(self.L)
        solve(self.a_mat, self.vort.vector(), L_mat, annotate=annotate)
        return self.vort
