from opentidalfarm import *
set_log_level(INFO)

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = SteadyConfiguration("mesh.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# set some genetic optimisation options
population_size = 60
max_iterations = 1000
number_of_turbines = 8
flow = helpers.get_ambient_flow(config)
# these are the remaining options which have defualt values
wake_model_type = "ApproximateShallowWater"
wake_model_parameters = None
survival_rate = 0.7
crossover_type = "uniform"
mutation_type = "fitness_proportionate"
selection_type = "best"
mutation_probability = 0.07
selection_options = None
options = {"jump_start": False,
           "disp": True,
           "update_every": 10,
           "predict_time": True,
           "save_stats": True,
           "save_fitness": True,
           "save_every": 1,
           "stats_file": "optimisation_stats.csv",
           "fitness_file": "optimisation_fitness.csv",
           "plot": True,
           "plot_file": "population_fitness.png"}

# initialize a genetic problem
problem = GeneticOptimisation(config, 
                              population_size,
                              max_iterations,
                              number_of_turbines,
                              flow, 
                              wake_model_type=wake_model_type, 
                              wake_model_parameters=wake_model_parameters,
                              survival_rate=survival_rate,
                              crossover_type=crossover_type,
                              mutation_type=mutation_type,
                              selection_type=selection_type,
                              mutation_probability=mutation_probability,
                              selection_options=selection_options,
                              options=options)

# optimize the problem
optimized_turbines = maximize_genetic(problem)

# its often good to then check the power of the best state using the full
# shallow water model
config.set_turbine_pos(optimized_turbines)
rf = ReducedFunctional(config)
rf.j(rf.initial_control())
