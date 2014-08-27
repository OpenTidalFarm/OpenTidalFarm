import numpy
from dolfin import *


class FunctionalIntegrator(object):

    def __init__(self, functional, final_only):
        self.final_only = final_only
        self.functional = functional
        self.vals = []
        self.times = []

    def add(self, time, state, tf, is_final):
        # FIXME: is_final flag seems to be computed wrongly in the shallow water
        # code
        if True or not self.final_only or (self.final_only and is_final):
            val = assemble(self.functional.Jt(state, tf))

            self.vals.append(val)
            self.times.append(time)

    def integrate(self, degree=1):
        if len(self.vals) == 0:
            raise ValueError("Cannot integrate empty set.")

        if self.final_only:
            return self.vals[-1]

        # FIXME: Don't assume constante timesteps
        dt = self.times[1]-self.times[0]

        # FIXME: Remove this, it just does not make any sense
        if degree == 0:
            dt = 1

        # Compute quadrature weights
        quads = numpy.ones(len(self.vals))

        if degree == 0:
            quads[0] = 0
        elif degree == 1:
            quads[0] = 0.5
            quads[-1] = 0.5
        else:
            raise ValueError("Unknown integration degree.")

        return sum(quads*dt*self.vals)
