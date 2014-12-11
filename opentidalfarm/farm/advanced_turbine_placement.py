import numpy as np
import opentidalfarm
import dolfin
from copy import deepcopy
#API notes - should pass in 'size farm' or 'greedy farm' as a prob_param then 
#continue as before throught the API. In reduced functional(?) we check if this
#parameter is switched on and if so we place the turbines from there - hopefully
#therefore having imported opentidalfarm already and not needing to import it as
#above.

#Each needs to be its own class! based on a base class which has useful things
#like parameters and access to the wake models etc!


class AdvancedTurbinePlacementParameters(object):

    def __init__(self):
        self.no_of_turbines = None

#    things that arent already packed in the rf
#    wake model to be used
#    dynamic array sizing (bool)


class AdvancedTurbinePlacement(object):
    """ A set of advanced turbine placement options
    """
    def __init__(self, reduced_functional):

	if not isinstance(reduced_functional, 
                opentidalfarm.reduced_functional_prototype.ReducedFunctionalPrototype):
	    raise TypeError, "reduced_functional must be of type ReducedFunctional"
	
	# Make some parameters more accessible
        self.rf = reduced_functional
	self.problem_params = reduced_functional._problem_params
	self.farm = reduced_functional._problem_params.tidal_farm
        self.solver = reduced_functional.solver

    def greedy_turbine_layout(self):
        """Arranges turbines greedily - solves the ambient flow field and
        positions turbines at the points of highest velocity - in descending
        order.

        TODO:   Only works for steady state
                Recalc ambient with wake model as each turbine is placed
                Assumes default CoupledSWSolver parameters

        :param num: Defines the number of turbines to add to the farm.
        :type turbine: Integer
        :param prob_params: The problem parameters
        :type prob_params: SteadySWProblem parameters object
        :param site_x_start: The minimum x-coordinate of the site.
        :type site_x_start: float
        :param site_x_end: The maximum x-coordinate of the site.
        :type site_x_end: float
        :param site_y_start: The minimum y-coordinate of the site.
        :type site_y_start: float
        :param site_y_end: The maximum y-coordinate of the site.
        :type site_y_end: float
        :raises: ValueError

        """

        dolfin.info('Calculating the ambient flow...')
        problem = opentidalfarm.SteadySWProblem(self.problem_params)
        sol_params = opentidalfarm.CoupledSWSolver.default_parameters()
        solver = opentidalfarm.CoupledSWSolver(problem, sol_params) 
        
        # Create the grid of potential turbine placement based on min dist
        # constraints
        spacing = self.farm.turbine_specification.minimum_distance
        num = self.farm.turbine_placement_parameters.no_of_turbines
        site_x_start = self.problem_params.tidal_farm.site_x_start
        site_x_end = self.problem_params.tidal_farm.site_x_end
        site_y_start = self.problem_params.tidal_farm.site_y_start
        site_y_end = self.problem_params.tidal_farm.site_y_end
        coordinates = [(x*spacing+site_x_start, y*spacing+site_y_start) for x 
                        in range(int((site_x_end-site_x_start)/spacing)) for y in
                        range(int((site_y_end-site_y_start)/spacing))]

        # Determine the ambient flow at each of those points and attach them
        # together
        ambient_flow_at_points = {}
        solution = solver.solve()
        for i in solution:
            for j in range(len(coordinates)):
                flow_vel = np.sqrt((i['u'][0](coordinates[j]))**2 +
                           (i['u'][1](coordinates[j])**2))
                ambient_flow_at_points.update({flow_vel: coordinates[j]})

        # Sort the coordinates by ambient velocity and extract the best however
        # many we need
        best_velocities = sorted(ambient_flow_at_points.keys())[-num:]

        # Add the turbines at those locations
        dolfin.info('Placing the turbines greedily...')
        for i in best_velocities:
            print ambient_flow_at_points[i]
            self.farm.add_turbine(ambient_flow_at_points[i]) 

    def sw_greedy_turbine_layout(self):
        """ Place the turbines greedily with shallow water solves

        The ambient flow is calculated, the first turbine is placed at the point
        of highest flow velocity, the flow is re-solved and the process is
        repeated with the next turbine.

        :param reduced_functional: Reduced functional which contains necessary
        parameters.
        :type reduced functional: An instance of the reduced_functional object.
        """
        dolfin.info("Placing turbines greedily... After each turbine is placed"
                    " the flow will be re-solved - for large numbers of turbines"
                    " this may take too long")
        num = self.farm.turbine_placement_parameters.no_of_turbines
        for i in range(num):
            dolfin.info('Calculating the flow to place turbine %i' % i)
            # Calculate the ambient flow 
            problem = opentidalfarm.SteadySWProblem(self.problem_params)
            sol_params = opentidalfarm.CoupledSWSolver.default_parameters()
            solver = opentidalfarm.CoupledSWSolver(problem, sol_params) 
            
            # Create the grid of potential turbine placement based on min dist
            # constraints
            print 'Advanced minimum distance: ', self.farm.turbine_specification.minimum_distance
            spacing = self.farm.turbine_specification.minimum_distance+5.
            print 'Set the spacing to: ', spacing
            num = self.farm.turbine_placement_parameters.no_of_turbines
            site_x_start = self.problem_params.tidal_farm.site_x_start
            site_x_end = self.problem_params.tidal_farm.site_x_end
            site_y_start = self.problem_params.tidal_farm.site_y_start
            site_y_end = self.problem_params.tidal_farm.site_y_end
            coordinates = [(x*spacing+site_x_start, y*spacing+site_y_start) for x 
                            in range(int((site_x_end-site_x_start)/spacing)+1) for y in
                            range(int((site_y_end-site_y_start)/spacing+1))]

            # Determine the ambient flow at each of those points and attach them
            # together 
            ambient_flow_at_points = {}
            solution = solver.solve(annotate=False) 
            state = solution.next()
            state = solution.next()
            u = state['state'] 
            for j in range(len(coordinates)):
                flow_vel = np.sqrt((u[0](coordinates[j]))**2 +
                                   (u[1](coordinates[j])**2))
                ambient_flow_at_points.update({flow_vel: coordinates[j]})
            
            # export pvd (TODO tidy up)
            #filename = 'with_%i_turbines.pvd' % i
            #pvd_file = dolfin.File(filename)
            #pvd_file << u[0]

            # Sort the coordinates by ambient velocity and extract the best 
            best_velocity = sorted(ambient_flow_at_points.keys())[-1]

            # Add the turbines at those locations
            dolfin.info('Current turbines at: ')
            print self.farm.turbine_positions
            dolfin.info('Placing turbine number %i' % i)
            self.farm.add_turbine(ambient_flow_at_points[best_velocity]) 
            self.farm.update() 
#            from IPython import embed; embed()


    def hybrid_greedy_turbine_layout(self):
        """ Place the turbines greedily based on a fitness function calculated
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

        def place_next_turbine(state, placed_turbines, grid):
            """ Use analytical wake model to determine the placement of the next
            turbine
            """

            def fitness_function(turbine_positions, wake_functional,
                                 wake_solver, wake_farm):
                """Determine the performance of the current turbine positions"""
                print turbine_positions
                position_tuples = turbine_positions.flatten().reshape(len(turbine_positions), 2) 
                flow_velocity = wake_solver.solve(position_tuples)
                fitness_function = wake_functional.power(flow_velocity) 
                return fitness_function

            def flow_field(position):
                """Extract just the flow field from the current state at
                'position'."""
                return self.u(position)[:-1]

            # We define a disposible 'wake' set up for this exercise
            wake_farm = self.farm            # deepcopy(self.farm)
            wake_prob_params = opentidalfarm.SteadyWakeProblem.default_parameters()
            wake_prob_params.domain = self.solver.problem.parameters.domain
            # Get the default paramteres from the Jensen model.
            model_params = opentidalfarm.Jensen.default_parameters()
            model_params.thrust_coefficient = 0.9
            model_params.turbine_radius = wake_farm.turbine_specification.radius
            wake_prob_params.wake_model = opentidalfarm.Jensen(model_params, flow_field)
            wake_prob_params.combination_model = opentidalfarm.GeometricSum
            wake_prob_params.tidal_farm = wake_farm
            # Now we can create the steady wake problem
            wake_problem = opentidalfarm.SteadyWakeProblem(wake_prob_params)
            wake_problem.parameters.tidal_farm.update()
            # Next we create a wake model solver.
            wake_solver = opentidalfarm.SteadyWakeSolver(wake_problem)
            # The wake model cannot currently use the generic prototype
            # functionals and must
            # use the WakePowerFunctional class.
            wake_functional = opentidalfarm.WakePowerFunctional(wake_farm) 
            # Start trialling the potential points
            trialled_points = {}
            for i in grid:
                if i not in placed_turbines:
                    existing_turbines = deepcopy(placed_turbines)
                    existing_turbines.append(i)
                    trialled_points.update({fitness_function(np.array(existing_turbines), wake_functional, wake_solver, wake_farm):i})
            best_fitness = sorted(trialled_points.keys())[-1]
            best_location = trialled_points[best_fitness]
            return best_location
        
        # Calculate the gird of possible turbine positions based upon the
        # minimum distance constraints 
        spacing = self.farm.turbine_specification.minimum_distance+5.  
        site_x_start = self.problem_params.tidal_farm.site_x_start
        site_x_end = self.problem_params.tidal_farm.site_x_end
        site_y_start = self.problem_params.tidal_farm.site_y_start
        site_y_end = self.problem_params.tidal_farm.site_y_end
        grid = [(x*spacing+site_x_start, y*spacing+site_y_start) for x 
                        in range(int((site_x_end-site_x_start)/spacing)+1) for y in
                        range(int((site_y_end-site_y_start)/spacing+1))]
        # Make number more accessible TODO put as an instance var 
        num = self.farm.turbine_placement_parameters.no_of_turbines
        for i in range(num):
            # Ensure this inner loop knows where turbines have been placed so
            # far in the outer model.
            placed_turbines = self.farm.turbine_positions
            dolfin.info('Calculating the flow to place turbine %i' % i)
            # Calculate the ambient flow 
            problem = opentidalfarm.SteadySWProblem(self.problem_params)
            sol_params = opentidalfarm.CoupledSWSolver.default_parameters()
            solver = opentidalfarm.CoupledSWSolver(problem, sol_params) 
            # Determine the ambient flow at each of those points and attach them
            # together 
            ambient_flow_at_points = {}
            solution = solver.solve(annotate=False) 
            state = solution.next()
            state = solution.next()
            u = state['state']
            self.u = u
            if i == 0:
                for j in range(len(grid)):
                    flow_vel = np.sqrt((u[0](grid[j]))**2 +
                                       (u[1](grid[j])**2))
                    ambient_flow_at_points.update({flow_vel: grid[j]}) 
                # Sort the coordinates by ambient velocity and extract the best 
                best_velocity = sorted(ambient_flow_at_points.keys())[-1]
                best_location = ambient_flow_at_points[best_velocity]
            else:
                best_location = place_next_turbine(u, placed_turbines, grid)

            # Add the turbines at those locations
            dolfin.info('Current turbines at: ')
            print self.farm.turbine_positions
            dolfin.info('Placing turbine number %i' % i)
#            from IPython import embed; embed()
            self.farm.add_turbine(best_location)
            self.farm.update() 











#class GreedyTurbinePlacement(object):
#    """ Places turbines into a farm greedily 
#    """##

#    def __init__(self):
#        self.turbine_positions = None#

#    def _discretize_rectangular_site

#    def _discretize_polygon_site

#    def _solve_flow

#    def _place_all_turbines

#    def _place_next_turbine


#class SizeArray(object)
#    """ Keeps placing turbines into the farm greedily until the functional 
#    drops
#    """
#    based on GTP above = intialises once then updates the turbine_positions
#    variable each time (will need to build a check in above to ensure those points
#    are eliminated

#    def __init__(self)
#        self.reduced_functional = reduced_functional
#        self.solver = reduced_functional.solver
#        self.functional = reduced_functional.functional
#        self.problem = reduced_functional.problem
#
#    1. Solve flow
#    2. Add a turbine greedily, compute functional
#    3. Add a turbine greedily, compute functional
#    4. for new_functional > old_functional:
#    5.    Add a turbine greedily, compute functional
#    6. Do a proper flow solve then try to add another?
#    7. Pass the initial layout back for optimisation
    
