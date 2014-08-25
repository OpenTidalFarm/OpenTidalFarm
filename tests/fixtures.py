import pytest
from opentidalfarm import *

@pytest.fixture
def sw_problem_parameters():
    # Set the parameters for the Shallow water problem
    parameters = ShallowWaterProblem.default_parameters()

    # Temporal settings
    parameters.start_time = Constant(0)
    parameters.finish_time = Constant(10)
    parameters.dt = Constant(0.1)

    # Use Crank-Nicolson to get a second-order time-scheme
    parameters.theta = Constant(0.6)

    # Activate the relevant terms
    parameters.include_advection = False
    parameters.include_viscosity = False
    parameters.linear_divergence = True

    # Physical settings
    parameters.friction = Constant(0.0)
    parameters.viscosity = Constant(0.0)
    parameters.depth = Constant(50)
    parameters.g = Constant(9.81)

    # Set the analytical boundary conditions
    parameters.bctype = "flather"
    parameters.flather_bc_expr = None

    parameters.functional_final_time_only = False

    return parameters
