import pytest
from opentidalfarm import *

# Based on UnsteadyConfiguration
@pytest.fixture
def steady_sw_problem_parameters():

    # Set the parameters for the Shallow water problem
    parameters = SteadySWProblem.default_parameters()

    # Activate the relevant terms
    parameters.include_advection = True
    parameters.include_viscosity = True
    parameters.linear_divergence = False

    # Physical settings
    parameters.friction = Constant(0.0025)
    parameters.viscosity = Constant(3.0)
    parameters.depth = Constant(50)
    parameters.g = Constant(9.81)

    return parameters

# Based on UnsteadyConfiguration
@pytest.fixture
def sw_nonlinear_problem_parameters():

    # Set the parameters for the Shallow water problem
    parameters = SWProblem.default_parameters()

    # Temporal settings
    period = 12. * 60 * 60
    parameters.start_time = Constant(1. / 4 * period)
    parameters.finish_time = Constant(5. / 4 * period)
    parameters.dt = Constant(period / 50)

    # Use Crank-Nicolson to get a second-order time-scheme
    parameters.theta = 1.0

    # Activate the relevant terms
    parameters.include_advection = True
    parameters.include_viscosity = True
    parameters.linear_divergence = False

    # Physical settings
    parameters.friction = Constant(0.0025)
    parameters.viscosity = Constant(3.0)
    parameters.depth = Constant(50)
    parameters.g = Constant(9.81)

    parameters.functional_final_time_only = False

    return parameters

# Based on UnsteadyConfiguration
@pytest.fixture
def multi_steady_sw_problem_parameters():

    # Set the parameters for the Shallow water problem
    parameters = MultiSteadySWProblem.default_parameters()

    # Temporal settings
    period = 12. * 60 * 60
    parameters.start_time = Constant(1. / 4 * period)
    parameters.finish_time = Constant(5. / 4 * period)
    parameters.dt = Constant(period / 50)

    # Activate the relevant terms
    parameters.include_advection = True
    parameters.include_viscosity = True
    parameters.linear_divergence = False

    # Physical settings
    parameters.friction = Constant(0.0025)
    parameters.viscosity = Constant(3.0)
    parameters.depth = Constant(50)
    parameters.g = Constant(9.81)

    parameters.functional_final_time_only = False

    return parameters

# Based on DefaultConfiguration
@pytest.fixture
def sw_linear_problem_parameters():
    # Set the parameters for the Shallow water problem
    parameters = SWProblem.default_parameters()

    # Temporal settings
    parameters.start_time = Constant(0)
    parameters.finish_time = Constant(100)
    parameters.dt = Constant(0.025)

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

    parameters.functional_final_time_only = False

    return parameters
