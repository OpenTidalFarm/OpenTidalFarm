''' This test checks the correctness of the functional gradient 
    with respect to the turbine position '''
import numpy
from opentidalfarm import *

def model(controls):
    config = DefaultConfiguration(nx=30, ny=10)
    domain = domains.RectangularDomain(3000, 1000, 5, 5)
    config.set_domain(domain)
  
    # Temporal settings
    period = 1.24*60*60 # Wave period
    config.params["start_time"] = period/4
    config.params["dt"] = period/50
    config.params["finish_time"] = config.params["start_time"] + \
            2*config.params["dt"]
    config.params["theta"] = 0.5
    config.params["include_advection"] = True 
    config.params["include_viscosity"] = True 
    config.params["viscosity"] = 20.0
    config.params["controls"] = controls
    config.params["dump_period"] = -1
    config.params["output_turbine_power"] = False
  
    # Boundary condition settings
    config.params["bctype"] = "strong_dirichlet"
  
    k = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
    eta0 = 2
    expression = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", 
        "0"), 
        eta0=eta0, 
        g=config.params["g"], 
        depth=config.params["depth"], 
        t=config.params["current_time"], 
        k=k)
  
    bc = DirichletBCSet(config)
    bc.add_analytic_u(1, expression)
    bc.add_analytic_u(2, expression)
    bc.add_analytic_u(3, expression)
    config.params["strong_bc"] = bc
  
    # Initial condition
    config.params["initial_condition"] = SinusoidalInitialCondition(config, eta0, 
            k, config.params["depth"])
  
  
    # Turbine settings
    config.params["friction"] = 0.0025
    # The turbine position is the control variable 
    config.params["turbine_pos"] = [[1000, 400], [2000, 600]] 
    # Choosing a friction coefficient of > 0.02 ensures that 
    # overlapping turbines lead to less power output.
    nb_turbines = len(config.params["turbine_pos"])
    config.params["turbine_friction"] = 0.2*numpy.ones(nb_turbines)
    config.params["turbine_x"] = 400
    config.params["turbine_y"] = 400
  
    rf = ReducedFunctional(config)
    return rf

class TestDiscreteTurbine(object):

    def test_gradient_of_peak_friction_passes_taylor_test(self):
        rf = model(controls=["turbine_pos"])
        m0 = rf.initial_control()

        p = numpy.random.rand(len(m0))
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.1, 
                perturbation_direction=p, number_of_tests=4)

        assert minconv > 1.98

    def test_gradient_of_position_passes_taylor_test(self):
        rf = model(controls=["turbine_friction"])
        m0 = rf.initial_control()

        p = numpy.random.rand(len(m0))
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.1, 
                perturbation_direction=p, number_of_tests=4)

        assert minconv > 1.98
