from opentidalfarm import *
import numpy

# Create a rectangular domain.
domain = FileDomain("mesh/mesh.xml")

def flow(x):
    """A dummy flow function where the flow is (1., 0.) everywhere."""
    return numpy.array([1.0, 0.0])

# Set the shallow water parameters
prob_params = SteadyWakeProblem.default_parameters()
prob_params.domain = domain
# The dummy model simply halves the flow speed if a turbine is in the wake of
# another.
prob_params.wake_model = DummyWakeModel(flow)
prob_params.combination_model = GeometricSum

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
farm.add_regular_turbine_layout(num_x=8, num_y=4)

prob_params.tidal_farm = farm

# Now we can create the steady wake problem
problem = SteadyWakeProblem(prob_params)

# Next we create a wake model solver.
solver = SteadyWakeSolver(problem)

# The wake model cannot currently use the generic prototype functionals and must
# use the WakePowerFunctional class.
functional = WakePowerFunctional(farm)

# The power is calculated by calling the 'power' method of the function with a
# list of velocities to calculate the power for. At present this power is by no
# means accurate (we simply take the cube of the flow velocity at each of the
# turbines).
flow_velocities = solver.solve()
print functional.power(flow_velocities)
