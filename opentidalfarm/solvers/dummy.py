from dolfin import *
from dolfin_adjoint import *
from solver import Solver
from ..problems import DummyProblem
from ..helpers import FrozenClass

class DummySolverParameters(FrozenClass):
    dump_period = -1
    print_individual_turbine_power = False

class DummySolver(Solver):

    def __init__(self, problem):

        if not isinstance(problem, DummyProblem):
            raise TypeError, "problem must be of type DummyProblem"

        self.problem = problem
        self.parameters = DummySolverParameters()
        self.tf = None
        self.state = None
        self.optimisation_iteration = 0

    def setup(self, W, turbine_field, annotate=True):
        (v, q) = TestFunctions(W)
        (u, h) = TrialFunctions(W)

        # Mass matrices
        self.M = inner(v, u) * dx
        self.M += inner(q, h) * dx

        self.A = (1.0 + turbine_field) * inner(v, u) * dx
        self.A += inner(q, h) * dx

        self.tf = Function(turbine_field, name="turbine_friction", annotate=annotate)

        self.annotate = annotate

    def solve(self, annotate=True):
        '''Solve (1+turbine)*M*state = M*old_state.
           The solution is a x-velocity of old_state/(turbine_friction + 1) and a zero pressure value y-velocity.
        '''

        problem_params = self.problem.parameters
        farm = problem_params.tidal_farm
        turbine_friction = farm.turbine_cache["turbine_field"]
        mesh = problem_params.domain.mesh

        # Create function spaces
        V, H = problem_params.finite_element(mesh)
        W = MixedFunctionSpace([V, H])

        # Load initial condition
        # Projection is necessary to obtain 2nd order convergence
        ic = project(problem_params.initial_condition, W)

        # Define functions
        state = Function(W, name="Current_state")  # solution of the next timestep
        self.state = state
        state.assign(ic, annotate=False)

        self.setup(W, turbine_friction, annotate)

        yield({"time": Constant(0),
               "u": state[0],
               "eta": state[1],
               "tf": self.tf,
               "state": state,
               "is_final": False})

        adjointer.time.start(0.0)
        tmpstate = Function(state.function_space(), name="tmp_state")
        rhs = action(self.M, state)
        # Solve the mini model
        solver_parameters = {"linear_solver": "cg", "preconditioner": "sor"}
        solve(self.A == rhs, tmpstate, solver_parameters=solver_parameters, annotate=self.annotate)

        state.assign(tmpstate, annotate=self.annotate)

        # Bump timestep to shut up libadjoint.
        adj_inc_timestep(time=float(self.problem.parameters.dt), finished=True)

        yield({"time": self.problem.parameters.dt,
               "u": state[0],
               "eta": state[1],
               "tf": self.tf,
               "state": state,
               "is_final": True})
