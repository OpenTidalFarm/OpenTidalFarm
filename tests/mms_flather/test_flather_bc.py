from opentidalfarm import *
from dolfin import log, INFO
from mms import compute_error, setup_model
import math


class TestFlatherBoundaryConditions(object):

    def test_spatial_convergence_is_two(self, sw_linear_problem_parameters,
            sin_ic):
        # Compute the errors
        errors = []
        levels = 3
        finish_time = 5.

        for l in range(levels):

            mesh_x = 4 * 2**l
            mesh_y = 2
            time_step = 0.25

            model = setup_model(sw_linear_problem_parameters, sin_ic, time_step, 
                                finish_time, mesh_x, mesh_y)
            error = compute_error(*model)
            errors.append(error)

        # Compute the orders of convergence
        conv = []
        for i in range(len(errors)-1):
            convergence_order = -math.log(errors[i+1]/errors[i], 2)
            conv.append(convergence_order)

        # Check the minimum convergence order and exit
        log(INFO, "Spatial order of convergence (expecting 2.0): %s." % conv)

        assert min(conv) > 1.9


    def test_temporal_convergence_is_two(self, sw_linear_problem_parameters,
            sin_ic):
        # Compute the errors
        errors = []
        levels = 4
        finish_time = 100.

        for l in range(levels):

            mesh_x = 2**4
            time_step = finish_time/(2*2**l)

            model = setup_model(sw_linear_problem_parameters, sin_ic, time_step, 
                                finish_time, mesh_x)
            error = compute_error(*model)
            errors.append(error)

        # Compute the orders of convergence 
        conv = [] 
        for i in range(len(errors)-1):
            convergence_order = -math.log(errors[i+1]/errors[i], 2)
            conv.append(convergence_order)

        # Check the minimum convergence order and exit
        log(INFO, "Temporal order of convergence (expecting 2.0): %s." % conv)

        assert min(conv) > 1.9
