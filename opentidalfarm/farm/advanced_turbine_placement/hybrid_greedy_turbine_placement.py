import dolfin
import numpy
import opentidalfarm as otf
from copy import deepcopy
from .prototype_advanced_placement import PrototypeAdvancedTurbinePlacement

class HybridGreedyTurbinePlacement(PrototypeAdvancedTurbinePlacement):
    """Place the turbines greedily based on a fitness function calculated
    as the power from an analytical wake model with shallow water solves in
    between turbine placements.

    The ambient flow is calculated, the first turbine is placed at the point
    of highest flow velocity, the flow is then re-solved. An analytical
    wake model is then used to determine the best place for the next
    turbine. The process is repeated with the next turbine.

    :param reduced_functional: Reduced functional which contains necessary
    parameters.
    :type reduced functional: An instance of the reduced_functional object.
    """

    dolfin.info("Placing turbines greedily... After each turbine is placed"
                " the flow will be re-solved - for large numbers of turbines"
                " this may take too long") 

    def __init__(self, reduced_functional):
        """ Constructor for the class
        """
        # Initialise the base_class
        super(HybridGreedyTurbinePlacement, self).__init__(reduced_functional)

        # We're immediately going to want the ambient flow
        self.find_ambient_flow_field()


    def best_point_from_wake_model(self):
        """ Use analytical wake model to determine the suitability for points to
        place the next turbine
        """

        def fitness_function(turbine_positions, wake_functional,
                                     wake_solver, wake_farm):
            """Determine the performance of the current turbine positions""" 
            position_tuples = turbine_positions.flatten().reshape(len(turbine_positions), 2)
            flow_velocity = wake_solver.solve(position_tuples)
            fitness_function = wake_functional.power(flow_velocity)
            return fitness_function

        def flow_field(position):
            """Extract just the flow field from the current state at
            'position'."""
            return self.ambient_state(position)[:-1]

        # We define a disposible 'wake' set up for this exercise
        wake_farm = self.farm
        wake_prob_params = otf.SteadyWakeProblem.default_parameters()
        wake_prob_params.domain = self.problem_params.domain
        # Get the default paramteres from the Jensen model.
        model_params = otf.Jensen.default_parameters() 
        model_params.turbine_radius = wake_farm.turbine_specification.radius
        wake_prob_params.wake_model = otf.Jensen(model_params, flow_field)
        wake_prob_params.combination_model = otf.GeometricSum
        wake_prob_params.tidal_farm = wake_farm
        # Now we can create the steady wake problem
        wake_problem = otf.SteadyWakeProblem(wake_prob_params)
        wake_problem.parameters.tidal_farm.update()
        # Next we create a wake model solver.
        wake_solver = otf.SteadyWakeSolver(wake_problem)
        wake_functional = otf.WakePowerFunctional(wake_farm)
        # Start trialling the potential points
        placed_turbines = self.farm.turbine_positions
        trialled_points = {}
        for i in self.grid:
            if i not in placed_turbines:
                existing_turbines = deepcopy(placed_turbines)
                existing_turbines.append(i)
                trialled_points.update({fitness_function(numpy.array(existing_turbines),
                                        wake_functional, wake_solver, wake_farm):i})
        dictionary = trialled_points
        for item in sorted(trialled_points):
            print item, trialled_points[item]
        best_performance = sorted(dictionary.keys())[-1]
        print dictionary[best_performance]
        return trialled_points

 
    def place(self):
        #if not self.advanced_placement_parameters.auto_size_array:
        dolfin.info('Calculating the flow to place turbine 1')
        self.find_ambient_velocity_on_grid()
        best_point = self.find_best_point(self.ambient_velocity_on_grid)
        self.place_turbine(best_point, 1)
        for i in range(self.placement_parameters.number_of_turbines-1):
            dolfin.info('Calculating the flow to place turbine %i' % (i+2))
            self.find_ambient_flow_field()
            best_point = self.find_best_point(self.best_point_from_wake_model())
            self.place_turbine(best_point, i+1)

