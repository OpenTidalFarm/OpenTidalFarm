from opentidalfarm import *
import pygenopt
import numpy

# Create a rectangular domain.
domain = FileDomain("mesh/mesh.xml")

# Create a shallow water problem first so we can calculate the flow through the
# domain with zero turbines.
shallow_water_problem_parameters = SteadySWProblem.default_parameters()
shallow_water_problem_parameters.domain = domain
shallow_water_problem_parameters.viscosity = Constant(3)
shallow_water_problem_parameters.depth = Constant(50)
shallow_water_problem_parameters.friction = Constant(0.0025)

# Specify the boundary conditions.
bcs = BoundaryConditionSet()
bcs.add_bc("u", Constant((2, 0)), facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)
# The free-slip boundary conditions.
bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="weak_dirichlet")

# Set the boundary conditions in the problem parameters.
shallow_water_problem_parameters.bcs = bcs

# Create the actual shallow water problem.
shallow_water_problem = SteadySWProblem(shallow_water_problem_parameters)

info("Calculating the ambient flow field (i.e. the flow field in an "
     "empty domain)")

# Set up the shallow water solver.
solver_parameters = CoupledSWSolver.default_parameters()
solver_parameters.dump_period = 1
sw_solver = CoupledSWSolver(shallow_water_problem, solver_parameters)
# Initialize the solver.
solver = sw_solver.solve()
state = solver.next()
# Solve the current state of the problem.
state = solver.next()

def flow_field(position):
    """Extract just the flow field from the current state at 'position'."""
    return state["state"](position)[:-1]

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

# Set the Wake problem.
prob_params = SteadyWakeProblem.default_parameters()
prob_params.domain = domain
# Get the default paramteres from the Jensen model.
model_params = Jensen.default_parameters()
model_params.turbine_radius = turbine.radius
prob_params.wake_model = Jensen(model_params, flow_field)
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

def fitness_function(turbine_positions):
    position_tuples = turbine_positions.reshape(len(turbine_positions)/2, 2)
    farm.set_turbine_positions(position_tuples)
    flow_velocity = solver.solve()
    return functional.power(flow_velocity)

number_of_turbines = 5

optimiser_parameters = pygenopt.Optimiser.default_parameters()
optimiser_parameters["fitness_function"] = fitness_function
optimiser_parameters["maximise"] = True
optimiser_parameters["generations"] = 200
optimiser_parameters["crossover"] = pygenopt.OnePoint
optimiser_parameters["mutator"] = pygenopt.FitnessProportionate
optimiser_parameters["mutation_rate"] = 0.07
optimiser_parameters["selection"] = pygenopt.Best
optimiser_parameters["selection_options"]["survival_rate"] = 0.7
optimiser_parameters["population_options"]["population_size"] = 50
optimiser_parameters["population_options"]["chromosome_shape"] = (
                                                    (number_of_turbines, 2))
optimiser_parameters["population_options"]["upper_limits"] = (640., 320.)
optimiser_parameters["population_options"]["lower_limits"] = (0., 0.)

optimiser = pygenopt.Optimiser(optimiser_parameters)
best = optimiser.optimise()
best = numpy.array(best).reshape(len(best)/2, 2)
farm.set_turbine_positions(best)

print farm.turbine_positions

visualise = pygenopt.Visualisation(optimiser)
visualise.each_generation_with_mean()
visualise.show()
