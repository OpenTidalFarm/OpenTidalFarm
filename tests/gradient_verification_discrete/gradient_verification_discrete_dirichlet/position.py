''' This test checks the correctness of the functional gradient 
    with respect to the turbine position '''
import sys
import numpy
from opentidalfarm import *
from dolfin import log, INFO, ERROR
from model import test_model

rf = test_model(controls=["turbine_pos"])
m0 = rf.initial_control()

p = numpy.random.rand(len(m0))
minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.1, 
        perturbation_direction=p, number_of_tests=4)

if minconv < 1.98:
    log(ERROR, "The gradient taylor remainder test failed.")
    sys.exit(1)
else:
    log(INFO, "Test passed")
