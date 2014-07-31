from opentidalfarm import *
from math import log
from mms import compute_error, setup_model
import sys
set_log_level(INFO)

if __name__ == "__main__":

    # Compute the errors
    errors = []
    levels = 4
    finish_time = 100.

    for l in range(levels):

        mesh_x = 2**4
        time_step = finish_time/(2*2**l)

        model = setup_model(time_step, finish_time, mesh_x)
        error = compute_error(*model)
        errors.append(error)

    # Compute the orders of convergence 
    conv = [] 
    for i in range(len(errors)-1):
        convergence_order = abs(log(errors[i+1]/errors[i], 2))
        conv.append(convergence_order)

    # Check the minimum convergence order and exit
    info("Temporal order of convergence (expecting 2.0): %s." % conv)

    if min(conv) < 1.9:
        info("Temporal convergence test failed for wave_flather")
        sys.exit(1)
    else:
        info("Test passed")
