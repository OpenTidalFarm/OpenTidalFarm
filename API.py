from opentidalfarm import *

# Create a prototype turbine
turbine = Turbine(diameter=20, minimum_distance=40)

# Initialise a farm
farm = RectangularFarm(site_x_start=1000, 
                       site_x_end=2000,
                       site_y_start=1000,
                       site_y_end=2000)

# Populate your farm with your turbines
farm.add_regular_turbine_layout(turbine, 
                                num_x = 10,
                                num_y = 10,
                                *additional args to locate the population within the farm)

# We create a problem, specify a solver and then define our objective
# Let's see that in action...

#################################################

# Grab default parameters for our problem
params = ShallowWaterProblem.default_parameters()

# Adapt them to our needs
params.mesh = Mesh("mesh.xml")    # Load the mesh from file
params.unsteady = True            # Time-dependent equations
params.start_time = 0.0           # Simulation start time
params.end_time = 0.0             # Simulation end time
bc = TidalForcing(...)            # Create forcing boundary conditions
params.boundary_conditions = [bc] # Set boundary conditions

# Aaaaand create our problem
problem = ShallowWaterProblem(parameters=params)

# Create a ShallowWaterSolver (we can tinker with the solver params by calling default_parameters() method 
# and adapting as above)
sw_solver = ShallowWaterSolver(problem) 

# Define the objectives that we want to use this solver-problem mix
power = Power(sw_solver, coefficient)                  # power.evaluate()
turbine_cost = TurbineCost(sw_solver, coefficient)     # cost.evaluate()

#################################################

# Let's say we want an objective requiring another solver...
# Let's see how that will work!

# Create our problem (let's just stick with the default parameters)
problem = CableRoutingProblem()

# Create a CableRoutingSolver
cc_solver = CableRoutingSolver(problem)

# Define the objectives that we want to use this solver-problem mix
cable_cost = CableCost(cc_solver, coefficient)

#################################################

# Specify our controls
controls = Controls(turbine_locations=True)

#################################################

# Define our objectives
objective = Objective(controls)       # or if you're just doing (e.g.) power you can do >> objective = Objective(controls, power=power) 
objective.power = power
objective.turbine_cost = turbine_cost
objective.cable_cost = cable_cost

# Create the reduced functional (in this case, this will be a combination of a shallow_water_reduced_functional 
# and a cable_cost_reduced_functional - but the user shouldn't have to understand the maths behind the rf so the
# ReducedFunctional class will work out what is required and assemble itself)
rf = ReducedFunctional(objective)

#################################################

site_bounds = farm.site_boundary_constraints()
turbine_proximity_constraint = farm.minimum_distance_constraint(controls)

maximize(rf, bounds=site_bounds, constraints=turbine_proximity_constraint, method="SLSQP", options={"maxiter": 200})
