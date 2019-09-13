import sys
import os
import pytest
import dolfin
import dolfin_adjoint
import opentidalfarm
from pyadjoint.tape import Tape, set_working_tape
from fixtures import sw_linear_problem_parameters
from fixtures import sw_nonlinear_problem_parameters
from fixtures import steady_sw_problem_parameters
from fixtures import multi_steady_sw_problem_parameters
from fixtures import sin_ic

default_params = dolfin.parameters.copy()
def pytest_runtest_setup(item):
    """ Hook function which is called before every test """

    # Reset dolfin parameter dictionary
    dolfin.parameters.update(default_params)

    # Reset adjoint state
#     dolfin_adjoint.adj_reset()
    set_working_tape(Tape())
