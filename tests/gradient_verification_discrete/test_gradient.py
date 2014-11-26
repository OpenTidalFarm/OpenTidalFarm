import numpy
from opentidalfarm import *

def model(controls, problem_params, sin_ic):
    domain = domains.RectangularDomain(0, 0, 3000, 1000, 5, 5)

    # Choosing a friction coefficient of > 0.02 ensures that
    # overlapping turbines lead to less power output.
    turbine = BumpTurbine(diameter=400., friction=0.2, controls=controls)

    # Create tidal farm
    farm = RectangularFarm(domain,
                           site_x_start=0,
                           site_x_end=3000,
                           site_y_start=0,
                           site_y_end=1000,
                           turbine=turbine)

    options["output_turbine_power"] = False
    # The turbine position is the control variable
    farm.add_turbine((1000,400))
    farm.add_turbine((2000,600))
    problem_params.tidal_farm = farm

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
        t=problem_params.start_time,
        k=k)
    bcs = BoundaryConditionSet()
    bcs.add_bc("u", expression, 1, "strong_dirichlet")
    bcs.add_bc("u", expression, 2, "strong_dirichlet")
    bcs.add_bc("u", expression, 3, "strong_dirichlet")
    problem_params.bcs = bcs

    # Initial condition
    problem_params.initial_condition = sin_ic(eta0,
            k, problem_params.depth, problem_params.start_time)

    # Physical parameters
    problem_params.friction = 0.0025

    # Create problem
    problem = SWProblem(problem_params)

    # Create solver
    solver_params = CoupledSWSolver.default_parameters()
    solver_params.dump_period = -1
    solver = CoupledSWSolver(problem, solver_params)

    functional = PowerFunctional(problem)
    control = TurbineFarmControl(farm)
    rf_params = ReducedFunctionalParameters()
    rf_params.automatic_scaling = False
    rf = ReducedFunctional(functional, control, solver, rf_params)
    return rf

class TestDiscreteTurbine(object):

    def test_gradient_of_peak_friction_passes_taylor_test(self,
            sw_linear_problem_parameters, sin_ic):
        rf = model(Controls(position=True), sw_linear_problem_parameters, sin_ic)
        m0 = rf.solver.problem.parameters.tidal_farm.control_array

        p = numpy.random.rand(len(m0))
        minconv = helpers.test_gradient_array(rf.evaluate, rf.derivative, m0,
                seed=0.1, perturbation_direction=p, number_of_tests=4)

        assert minconv > 1.97

    def test_gradient_of_position_passes_taylor_test(self,
            sw_linear_problem_parameters, sin_ic):
        rf = model(Controls(friction=True), sw_linear_problem_parameters, sin_ic)
        m0 = rf.solver.problem.parameters.tidal_farm.control_array

        p = numpy.random.rand(len(m0))
        minconv = helpers.test_gradient_array(rf.evaluate, rf.derivative, m0,
                seed=0.1, perturbation_direction=p, number_of_tests=4)

        assert minconv > 1.97
