from opentidalfarm import *
import numpy

# Create a rectangular domain.
# In this setup the wake model does not actually use the mesh, but the requires
# a domain which by necessesity has a mesh, thus the low number of mesh elements
# (10x10) does not matter.
x0 = 0.
y0 = 0.
x1 = 640.
y1 = 320.
nx = 10
ny = 10
domain = RectangularDomain(x0, y0, x1, y1, nx, ny)

# The next step is to create the turbine farm. In this case, the
# farm consists of 32 turbines, initially deployed in a regular grid layout.
# This layout will be the starting guess for the optimization.

# Before adding turbines we must specify the type of turbines used in the array.
# Here we used the default BumpTurbine which defaults to being controlled by
# just position. The diameter and friction are set. The minimum distance between
# turbines if not specified is set to 1.5*diameter.
turbine = BumpTurbine(diameter=20.0, friction=12.0)

# A rectangular farm is defined using the domain and the site dimensions.
farm = RectangularFarm(domain, site_x_start=160, site_x_end=480,
                       site_y_start=80, site_y_end=240, turbine=turbine)

# Turbines are then added to the site in a regular grid layout.
farm.add_regular_turbine_layout(num_x=2, num_y=2)


def flow(x):
    """A dummy flow function where the flow is (1., 0.) everywhere."""
    return numpy.array([2.0, 0.0])

# Set the Wake problem.
prob_params = SteadyWakeProblem.default_parameters()
prob_params.domain = domain
# Get the default paramteres from the Jensen model.
model_params = Jensen.default_parameters()
model_params.turbine_radius = turbine.radius
prob_params.wake_model = Jensen(model_params, flow)
# At present this is the only combination model that is correctly working.
prob_params.combination_model = GeometricSum
prob_params.tidal_farm = farm

# Now we can create the steady wake problem
problem = SteadyWakeProblem(prob_params)

# Next we create a wake model solver.
solver = SteadyWakeSolver(problem)

# The wake model cannot currently use the generic prototype functionals and must
# use the WakePowerFunctional class.
functional = WakePowerFunctional(farm)

# The power is calculated by calling the "power" method of the functional with a
# list of velocities to calculate the power for. At present this POWER DOES
# NOT REPRESENT AN ACCURATE POWER in watts! (We simply take the cube of the flow
# velocity at each of the turbines.)
flow_velocities = solver.solve()
print functional.power(flow_velocities)
