import numpy
from dolfin import *
from dolfin_adjoint import *
from ..problems import MultiSteadySWProblem


class TimeIntegrator(object):

    def __init__(self, problem, functional, final_only):
        self.problem = problem
        self.functional = functional
        self.final_only = final_only

        self.vals = []
        self.times = []

    def add(self, time, state, tf, is_final):
        if not self.final_only or (self.final_only and is_final):
            val = assemble(self.functional.Jt(state, tf))
            self.vals.append(val)
            self.times.append(time)

    def integrate(self):

        if len(self.vals) == 0:
            raise ValueError("Cannot integrate empty set.")

        if self.final_only:
            return self.vals[-1]

        # FIXME: Don't assume constant timesteps
        dt = self.times[1]-self.times[0]

        # Compute quadrature weights
        w = numpy.ones(len(self.times))

        # The multi-steady state case is special in that we want to integrate
        # over time, but without the initial guess.
        w[-1] = 0.0
        if type(self.problem) == MultiSteadySWProblem:
            w[0] = 0.
            w[1] = 0.5
        else:
            w[0] = 0.5
        w[-1] += 0.5

        return sum(w * dt * self.vals)

    def dolfin_adjoint_functional(self):
        # The functional depends on dolfin functions which are not in scope.
        # But dolfin-adjoint only cares about the name, hence it is sufficient
        # to create dummy functions with the appropriate names.
        R = FunctionSpace(self.problem.parameters.domain.mesh, "R", 0)
        Rvec = MixedFunctionSpace([VectorFunctionSpace(self.problem.parameters.domain.mesh, "R", 0), R])
        tf = Function(R, name="turbine_friction")
        state = Function(Rvec, name="Current_state")

        if self.final_only:
            return Functional(self.functional.Jt(state, tf) * dt[FINISH_TIME])

        if type(self.problem) == MultiSteadySWProblem:
            return Functional(self.functional.Jt(state, tf) *
                    dt[float(self.times[1]):])
        else:
            return Functional(self.functional.Jt(state, tf) * dt)
