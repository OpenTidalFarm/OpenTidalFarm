''' This test checks the performance of the turbine model implementation. One
of the slowest part of the code is the interpolation of the turbine field onto
a discrete function space, because turbine's eval function is called very
often.  This test was used to optimise the eval implementation. On 4 Intel(R)
Xeon(R) CPU  E5506  @ 2.13GHz the benchmark time should be around 11s. '''

from opentidalfarm import *
import numpy
import cProfile


def default_config():
    # We set the perturbation_direction with a constant seed,
    # so that it is consistent in a parallel environment.
    config = configuration.DefaultConfiguration(
        nx=600,
        ny=200,
        finite_element=finite_elements.p1dgp2
    )
    domain = domains.RectangularDomain(3000, 1000, 600, 200)
    config.set_domain(domain)

    period = 1.24*60*60  # Wave period
    config.params["k"] = 2*pi/(period*sqrt(config.params["g"] *
                               config.params["depth"]))
    config.params["finish_time"] = 2./4*period
    config.params["dt"] = config.params["finish_time"]/20
    config.params["dump_period"] = 1
    config.params["verbose"] = 100

    # Start at rest state
    config.params["start_time"] = config.params["finish_time"] - 3 * \
        config.params["dt"]  # period/4

    # Turbine settings
    config.params["friction"] = 0.0025
    # The turbine position is the control variable
    turbine_pos = []
    border = 100
    for x_r in numpy.linspace(0.+border, config.domain.basin_x-border, 30):
        for y_r in numpy.linspace(0.+border, config.domain.basin_y-border, 10):
            turbine_pos.append((float(x_r), float(y_r)))

    config.set_turbine_pos(turbine_pos, friction=1.0)
    info_blue("Deployed " + str(len(turbine_pos)) + " turbines.")

    config.params["turbine_x"] = 190  # We overlap the turbines on purpose
    config.params["turbine_y"] = 20

    return config

# Set up the turbine friction field using the provided control variable
config = default_config()
tf = Function(config.turbine_function_space)
turbines = Turbines(config.turbine_function_space, config.params)

# Benchmark the generation of the turbine function
cProfile.run("tf = turbines()")

print "norm(tf) = ", norm(tf)
correct_norm = 634.425772066
if abs(norm(tf) - correct_norm) > 0.000000001:
    log(ERROR, "Warning: Wrong norm. Should be %e" % correct_norm)
log(INFO, "This test should take round 1 min.")
