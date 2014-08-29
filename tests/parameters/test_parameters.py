from opentidalfarm import *
import pytest


def test_IfUnkownParameterIsSet_then_ExceptionIsRaised():

    param = SWProblem.default_parameters()
    try:
        param.depth = 10
    except Exception:
        py.test.fail("Writing parameters to existing attributes should be ok.")

    with pytest.raises(TypeError):
        param.xxx = 10
