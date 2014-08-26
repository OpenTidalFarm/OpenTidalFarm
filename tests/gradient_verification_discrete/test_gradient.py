''' This test checks the correctness of the functional gradient 
    with respect to the turbine position '''
import numpy
from opentidalfarm import *

def model(controls, problem_params):
    domain = domains.RectangularDomain(0, 0, 3000, 1000, 5, 5)

    # Turbine settings
    config = DefaultConfiguration(domain)
    config.params["controls"] = controls
    config.params["output_turbine_power"] = False
    # The turbine position is the control variable 
    config.params["turbine_pos"] = [[1000, 400], [2000, 600]] 
    # Choosing a friction coefficient of > 0.02 ensures that 
    # overlapping turbines lead to less power output.
    nb_turbines = len(config.params["turbine_pos"])
    config.params["turbine_friction"] = 0.2*numpy.ones(nb_turbines)
    config.params["turbine_x"] = 400
    config.params["turbine_y"] = 400

    # Domain
    problem_params.domain = domain
  
    # Temporal settings
    period = 1.24*60*60 # Wave period
    problem_params.start_time = Constant(period/4)
    problem_params.dt = Constant(period/50)
    problem_params.finish_time = Constant(problem_params.start_time + \
            2*problem_params.dt)
    problem_params.theta = Constant(0.5)
    problem_params.include_advection = True 
    problem_params.include_viscosity = True 
    problem_params.viscosity = 20.0
  
    # Boundary condition settings
    k = Constant(2*pi/(period*sqrt(problem_params.g*problem_params.depth)))
    eta0 = 2
    expression = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", 
        "0"), 
        eta0=eta0, 
        g=problem_params.g, 
        depth=problem_params.depth,
        t=problem_params.current_time, 
        k=k)
  
    bcs = BoundaryConditionSet()
    bcs.add_bc("u", expression, 1, "strong_dirichlet")
    bcs.add_bc("u", expression, 2, "strong_dirichlet")
    bcs.add_bc("u", expression, 3, "strong_dirichlet")
    problem_params.bcs = bcs
  
    # Initial condition
    problem_params.initial_condition = SinusoidalInitialCondition(eta0, 
            k, problem_params.depth, problem_params.start_time)
  
    # Physical parameters
    problem_params.friction = 0.0025

    problem = ShallowWaterProblem(problem_params)

    solver_params = ShallowWaterSolver.default_parameters()
    solver_params.dump_period = -1
  
    solver = ShallowWaterSolver(problem, solver_params, config)
    rf = ReducedFunctional(config, solver)
    return rf

class TestDiscreteTurbine(object):

    def test_gradient_of_peak_friction_passes_taylor_test(self,
            sw_linear_problem_parameters):
        rf = model(["turbine_pos"], sw_linear_problem_parameters)
        m0 = rf.initial_control()

        p = numpy.random.rand(len(m0))
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.1, 
                perturbation_direction=p, number_of_tests=4)

        assert minconv > 1.97

    def test_gradient_of_position_passes_taylor_test(self,
            sw_linear_problem_parameters):
        rf = model(["turbine_friction"], sw_linear_problem_parameters)
        m0 = rf.initial_control()

        p = numpy.random.rand(len(m0))
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.1, 
                perturbation_direction=p, number_of_tests=4)

        assert minconv > 1.97
