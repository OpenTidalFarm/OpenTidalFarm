from dolfin import *
from dolfin_adjoint import *


class MiniModel:
    def __init__(self, config, turbine_field, functional=None, annotate=True):
        (v, q) = TestFunctions(config.function_space)
        (u, h) = TrialFunctions(config.function_space)

        # Mass matrices
        self.M = inner(v, u) * dx
        self.M += inner(q, h) * dx

        self.A = (1.0 + turbine_field) * inner(v, u) * dx
        self.A += inner(q, h) * dx

        self.tf = Function(turbine_field, name="turbine_friction", annotate=annotate)

        self.annotate = annotate
        self.functional = functional
        self.config = config

    def __call__(self, state):
        '''Solve (1+turbine)*M*state = M*old_state.
           The solution is a x-velocity of old_state/(turbine_friction + 1) and a zero pressure value y-velocity.
        '''

        params = self.config.params
        if functional is not None and not params["functional_final_time_only"]:
            j = 0.5 * params["dt"] * assemble(self.functional.Jt(state, self.tf))

        adjointer.time.start(0.0)
        tmpstate = Function(state.function_space(), name="tmp_state")
        rhs = action(self.M, state)
        # Solve the mini model
        solver_parameters = {"linear_solver": "cg", "preconditioner": "sor"}
        solve(self.A == rhs, tmpstate, solver_parameters=solver_parameters, annotate=self.annotate)

        state.assign(tmpstate, annotate=self.annotate)

        # Bump timestep to shut up libadjoint.
        adj_inc_timestep(time=params["dt"], finished=True)

        if self.functional is not None:
            if params["functional_final_time_only"]:
                j = assemble(self.functional.Jt(state, self.tf))
            else:
                j += 0.5 * params["dt"] * assemble(self.functional.Jt(state, self.tf))
            return j


def mini_model_solve(config, state, turbine_field, functional=None, annotate=True, linear_solver="default", preconditioner="default", u_source=None):
    mini_model = MiniModel(config, turbine_field, functional, annotate)
    return mini_model(state)
