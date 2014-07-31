from opentidalfarm import *
from math import log
from mms import compute_error, setup_model
import sys
set_log_level(INFO)

if __name__ == "__main__":

    # Compute the errors
    errors = []
    levels = 3
    finish_time = 5.

    for l in range(levels):

        mesh_x = 4 * 2**l
        mesh_y = 2
        time_step = 0.25

        model = setup_model(time_step, finish_time, mesh_x, mesh_y)
        error = compute_error(*model)
        errors.append(error)

    # Compute the orders of convergence
    conv = []
    for i in range(len(errors)-1):
        convergence_order = abs(log(errors[i+1]/errors[i], 2))
        conv.append(convergence_order)

    # Check the minimum convergence order and exit
    info("Spatial order of convergence (expecting 2.0): %s." % conv)

    if min(conv) < 1.9:
        info("Spatial convergence test failed for wave_flather")
        sys.exit(1)
    else:
        info("Test passed")
