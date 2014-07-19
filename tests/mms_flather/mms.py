from opentidalfarm import *
from dolfin_adjoint import adj_reset
import opentidalfarm.domains

def compute_error(config, eta0, k):
    # Define the state function
    state = Function(config.function_space)
    state.interpolate(SinusoidalInitialCondition(config, eta0, k, config.params["depth"]))
  
    # Define the analytical (MMS) solution
    u_exact = "eta0*sqrt(g/depth) * cos(k*x[0]-sqrt(g*depth)*k*t)" 
    du_exact = "(- eta0*sqrt(g/depth) * sin(k*x[0]-sqrt(g*depth)*k*t) * k)"
    eta_exact = "eta0*cos(k*x[0]-sqrt(g*depth)*k*t)"
  
    # The source terms, which originate from plugging the analytical solution 
    # in the flow equations
    advection_source = u_exact + " * " + du_exact
    viscosity_source = "friction/depth * " + u_exact + " * pow(pow(" + u_exact + ", 2), 0.5)"
  
    source = Expression((viscosity_source + "+" + advection_source, "0.0"),
                        eta0=eta0, 
                        g=config.params["g"],
                        depth=config.params["depth"],
                        t=config.params["current_time"], 
                        k=k, 
                        friction=config.params["friction"])
  
    # Run the shallow water model
    shallow_water_model.sw_solve(config, state, annotate=False, u_source=source)
  
    # Compare the difference to the analytical solution
    analytic_sol = Expression((u_exact, "0", eta_exact),
                              eta0=eta0, g=config.params["g"],
                              depth=config.params["depth"], t=config.params["current_time"], k=k)
    exactstate = Function(config.function_space)
    exactstate.interpolate(analytic_sol)
    e = state - exactstate
    return sqrt(assemble(dot(e,e)*dx))

def setup_model(time_step, finish_time, mesh_x):
    # Reset the adjoint tape to keep dolfin-adjoint happy 
    adj_reset()

    # Define the mesh size. The solution is constant in the 
    # y-direction, a very coarse mesh is sufficient
    mesh_y = 2

    config = configuration.DefaultConfiguration(nx=mesh_x, 
                                                ny=mesh_y, 
                                                finite_element=finite_elements.p1dgp2)

    domain = opentidalfarm.domains.RectangularDomain(3000, 1000, 
                                                     mesh_x, mesh_y)

    config.set_domain(domain)
    eta0 = 2.0
    k = pi/config.domain.basin_x

    # Temporal settings
    config.params["finish_time"] = Constant(finish_time)
    config.params["dt"] = Constant(time_step)

    # Use Crank-Nicolson to get a second-order time-scheme
    config.params["theta"] = 0.5

    # Activate the relevant terms
    config.params["include_advection"] = True
    config.params["friction"] = 0.25 

    # Set the analytical boundary conditions
    config.params["flather_bc_expr"] = Expression(("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"), 
                                                   eta0=eta0, 
                                                   g=config.params["g"], 
                                                   depth=config.params["depth"], 
                                                   t=config.params["current_time"], 
                                                   k=k)

    config.params["dump_period"] = 100000

    return config, eta0, k


