import sys
import os
import pytest
import dolfin
import dolfin_adjoint
import opentidalfarm
from fixtures import sw_problem_parameters

# Automatically parallelize over all cpus
def pytest_cmdline_preparse(args):
    if 'xdist' in sys.modules: # pytest-xdist plugin
        import multiprocessing
        num = multiprocessing.cpu_count()
        args[:] = ["-n", str(num)] + args

del dolfin_adjoint.test_initial_condition_adjoint
del dolfin_adjoint.test_initial_condition_tlm
del dolfin_adjoint.test_scalar_parameters_adjoint
del dolfin_adjoint.test_initial_condition_adjoint_cdiff
del dolfin_adjoint.test_scalar_parameter_adjoint

default_params = dolfin.parameters.copy()
def pytest_runtest_setup(item):
    """ Hook function which is called before every test """

    # Reset dolfin parameter dictionary
    dolfin.parameters.update(default_params)

    # Reset adjoint state
    dolfin_adjoint.adj_reset()
