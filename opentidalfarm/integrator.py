import numpy
from dolfin import *
from dolfin_adjoint import *
from problems import MultiSteadyShallowWaterProblem


class FunctionalIntegrator(object):

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

    def integrate(self, degree=1):
        if len(self.vals) == 0:
            raise ValueError("Cannot integrate empty set.")

        print "*"*20
        print "vals", self.vals

        if self.final_only:
            return self.vals[-1]

        # FIXME: Don't assume constant timesteps
        dt = self.times[1]-self.times[0]

        # FIXME: Remove this, it just does not make any sense
        if degree == 0:
            dt = 1

        # Compute quadrature weights
        quads = numpy.ones(len(self.times))

        if degree == 0:
            quads[0] = 0
        elif degree == 1:
            if type(self.problem) == MultiSteadyShallowWaterProblem:
                quads[0] = 0.
                quads[1] = 0.5
            else:
                quads[0] = 0.5
            quads[-1] = 0.5
        else:
            raise ValueError("Unknown integration degree.")

        return sum(quads*dt*self.vals)

    def dolfin_adjoint_functional(self, degree):
        # The functional depends on dolfin functions which are not in scope.
        # But dolfin-adjoint only cares about the name, hence it is sufficient 
        # to create dummy functions with the appropriate names.
        R = FunctionSpace(self.problem.parameters.domain.mesh, "R", 0)
        Rvec = VectorFunctionSpace(self.problem.parameters.domain.mesh, "R", 0)
        tf = Function(R, name="turbine_friction")
        state = Function(Rvec, name="Current_state")

        if self.final_only:
            return Functional(self.functional.Jt(state, tf) * dt[FINISH_TIME])

        elif degree == 0:
            timesteps = list(self.times)

            from problems.shallow_water import ShallowWaterProblemParameters
            if not type(self.problem.parameters) is ShallowWaterProblemParameters:
                timesteps.pop(0)

            return Functional(sum(self.functional.Jt(state, tf) * dt[float(t)] for t in timesteps))

        else:
            if type(self.problem) == MultiSteadyShallowWaterProblem:
                return Functional(self.functional.Jt(state, tf) *
                        dt[float(self.times[1]):])
            else:
                return Functional(self.functional.Jt(state, tf) * dt)
