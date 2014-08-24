from dolfin import Constant
from dolfin_adjoint import Constant
from problem import Problem


class SteadyShallowWaterProblem(Problem):

    def __init__(self, parameters):
        self.parameters = parameters

    @property
    def is_transient(self):
        return False

    @staticmethod
    def default_parameters():
        ''' Returns a dictionary with the default parameters '''

        parameters = {}

        # Physical parameters
        parameters["depth"] = Constant(50.)
        parameters["g"] = Constant(9.81)
        parameters["viscosity"] = Constant(3.0)
        parameters["friction"] = Constant(0.0025)

        # Equation settings
        parameters["include_advection"] = True
        parameters["include_viscosity"] = True
        parameters["linear_divergence"] = False

        # Boundary conditions
        parameters["bctype"] = 'strong_dirichlet'
        parameters["strong_bc"] = None
        parameters["free_slip_on_sides"] = True
        parameters["eta_weak_dirichlet_bc_expr"] = None

        return parameters


class ShallowWaterProblem(Problem):

    def __init__(self, parameters):
        self.parameters = parameters

    @property
    def is_transient(self):
        return True

    @staticmethod
    def default_parameters():
        ''' Returns a dictionary with the default parameters '''

        parameters = SteadyShallowWaterProblem.default_parameters()

        # Time parameters
        parameters["theta"] = 1.0
        parameters["dt"] = 1.
        parameters["start_time"] = 0.0
        parameters["current_time"] = 0.0
        parameters["finish_time"] = 100.0
        parameters["t"] = 0.0

        # Equation settings
        parameters["include_time_term"] = True

        # Functional time integration parameters
        parameters["functional_final_time_only"] = True
        parameters["functional_quadrature_degree"] = 1

        return parameters
