from dolfin import *
from dolfin_adjoint import *
from solver import Solver
from ..problems import DummyProblem


class DummySolver(Solver):

    def __init__(self, problem, config):

        if not isinstance(problem, DummyProblem):
            raise TypeError, "problem must be of type DummyProblem"

        self.problem = problem
        self.config = config
        self.tf = None

    def setup(self, turbine_field, functional=None, annotate=True):
        (v, q) = TestFunctions(self.config.function_space)
        (u, h) = TrialFunctions(self.config.function_space)

        # Mass matrices
        self.M = inner(v, u) * dx
        self.M += inner(q, h) * dx

        self.A = (1.0 + turbine_field) * inner(v, u) * dx
        self.A += inner(q, h) * dx

        self.tf = Function(turbine_field, name="turbine_friction", annotate=annotate)

        self.annotate = annotate
        self.functional = functional

    def solve(self, state, turbine_field, functional=None, annotate=True,
            linear_solver="default", preconditioner="default", u_source=None):
        '''Solve (1+turbine)*M*state = M*old_state.
           The solution is a x-velocity of old_state/(turbine_friction + 1) and a zero pressure value y-velocity.
        '''

        self.setup(turbine_field, functional, annotate)

        if functional is not None and not self.problem.parameters.functional_final_time_only:
            j = 0.5 * assemble(self.problem.parameters.dt * self.functional.Jt(state, self.tf))

        adjointer.time.start(0.0)
        tmpstate = Function(state.function_space(), name="tmp_state")
        rhs = action(self.M, state)
        # Solve the mini model
        solver_parameters = {"linear_solver": "cg", "preconditioner": "sor"}
        solve(self.A == rhs, tmpstate, solver_parameters=solver_parameters, annotate=self.annotate)

        state.assign(tmpstate, annotate=self.annotate)

        # Bump timestep to shut up libadjoint.
        adj_inc_timestep(time=float(self.problem.parameters.dt), finished=True)

        if self.functional is not None:
            if self.problem.parameters.functional_final_time_only:
                j = assemble(self.functional.Jt(state, self.tf))
            else:
                j += 0.5 * assemble(self.problem.parameters.dt * self.functional.Jt(state, self.tf))
            return j
