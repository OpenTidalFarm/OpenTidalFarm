from opentidalfarm import *
from dolfin_adjoint import adj_reset


def compute_error(problem, eta0, k):

    # Define the analytical (MMS) solution
    u_exact = "eta0*sqrt(g/depth) * cos(k*x[0]-sqrt(g*depth)*k*t)"
    du_exact = "(- eta0*sqrt(g/depth) * sin(k*x[0]-sqrt(g*depth)*k*t) * k)"
    eta_exact = "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"

    # The source terms, which originate from plugging the analytical solution
    # in the flow equations
    advection_source = u_exact + " * " + du_exact
    viscosity_source = "friction/depth * " + u_exact + " * pow(pow(" + \
                       u_exact + ", 2), 0.5)"

    source = Expression((viscosity_source + "+" + advection_source, "0.0"),
                        eta0=eta0,
                        g=problem.parameters.g,
                        depth=problem.parameters.depth,
                        t=Constant(problem.parameters.start_time),
                        k=k,
                        friction=problem.parameters.friction)

    parameters = SWSolver.default_parameters()
    parameters.dump_period = -1
    solver = SWSolver(problem, parameters)

    solver.solve(annotate=False, u_source=source)

    for s in solver.solve(annotate=False, u_source=source):
        pass
    state = s["state"]

    # Compare the difference to the analytical solution
    analytic_sol = Expression((u_exact, "0", eta_exact),
                              eta0=eta0, g=problem.parameters.g,
                              depth=problem.parameters.depth,
                              t=Constant(problem.parameters.finish_time),
                              k=k)
    return errornorm(analytic_sol, state)


def setup_model(parameters, time_step, finish_time, mesh_x, mesh_y=2):
    # Note: The analytical solution is constant in the
    # y-direction, hence a coarse y-resolution is sufficient.

    # Reset the adjoint tape to keep dolfin-adjoint happy
    adj_reset()

    eta0 = 2.0
    k = pi/3000.

    # Temporal settings
    parameters.start_time = Constant(0)
    parameters.finish_time = Constant(finish_time)
    parameters.dt = Constant(time_step)

    # Finite element
    parameters.finite_element = finite_elements.p1dgp2

    # Use Crank-Nicolson to get a second-order time-scheme
    parameters.theta = Constant(0.5)

    # Activate the relevant terms
    parameters.include_advection = True
    parameters.include_viscosity = False   
    parameters.linear_divergence = True 

    # Physical settings
    parameters.friction = Constant(0.25)
    parameters.viscosity = Constant(0.0)
    parameters.domain = domains.RectangularDomain(0, 0, 3000, 1000, mesh_x, 
                                                  mesh_y)

    # Initial condition
    ic_expr = SinusoidalInitialCondition(eta0, k,
                                         parameters.depth, 
                                         parameters.start_time)
    parameters.initial_condition = ic_expr

    # Set the analytical boundary conditions
    flather_expr = Expression(
        ("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"), 
        eta0=eta0, 
        g=parameters.g, 
        depth=parameters.depth, 
        t=Constant(parameters.start_time),
        k=k)

    bcs = BoundaryConditionSet()
    bcs.add_bc("u", flather_expr, facet_id=[1, 2], bctype="flather")
    bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="weak_dirichlet")
    parameters.bcs = bcs

    problem = SWProblem(parameters)

    return problem, eta0, k
