import numpy as np
from opentidalfarm import *
from .. import reduced_functional_prototype 

#API notes - should pass in 'size farm' or 'greedy farm' as a prob_param then 
#continue as before throught the API. In reduced functional(?) we check if this
#parameter is switched on and if so we place the turbines from there - hopefully
#therefore having imported opentidalfarm already and not needing to import it as
#above.

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

	if not isinstance(reduced_functional, reduced_functional_prototype.ReducedFunctionalPrototype):
	    raise TypeError, "reduced_functional must be of type ReducedFunctional"
	
	# Make some parameters more accessible
        self.rf = reduced_functional
	self.problem_params = reduced_functional._problem_params
	self.farm = reduced_functional._problem_params.tidal_farm

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

        # Calculate the ambient flow
        from opentidalfarm import *
        problem = SteadySWProblem(self.problem_params)
        sol_params = CoupledSWSolver.default_parameters()
        solver = CoupledSWSolver(problem, sol_params) 
        
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
        for i in best_velocities:
            print ambient_flow_at_points[i]
            self.farm.add_turbine(ambient_flow_at_points[i]) 

    def _greedy_turbine_layout(self):
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

        # Calculate the ambient flow
        from opentidalfarm import *
        problem = SteadySWProblem(self.problem_params)
        sol_params = CoupledSWSolver.default_parameters()
        solver = CoupledSWSolver(problem, sol_params) 
        
        # Create the grid of potential turbine placement based on min dist
        # constraints
        spacing = self.farm.turbine_specification.minimum_distance
        num = self.farm.no_of_turbines
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
        for i in best_velocities:
            print ambient_flow_at_points[i]
            self.farm.add_turbine(ambient_flow_at_points[i]) 





    def __greedy_turbine_layout(self, num, prob_params, site_x_start, site_x_end,
                                site_y_start, site_y_end):
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
        import opentidalfarm
        import numpy as np
        # Calculate the ambient flow
        problem = opentidalfarm.SteadySWProblem(prob_params)
        sol_params = opentidalfarm.CoupledSWSolver.default_parameters()
        solver = opentidalfarm.CoupledSWSolver(problem, sol_params)
        # TODO it may be handy to save the ambient flow to file
        # ambient_flow = File("ambient_flow.pvd")
        solution = solver.solve()
        # Create the grid of potential turbine placemenst based on min dist
        # constraints
        spacing = 30 # TODO get this fromt the problem
        coordinates = [(x*spacing+site_x_start, y*spacing+site_y_start) for x 
                        in range((site_x_end-site_x_start)/spacing) for y in
                        range((site_y_end-site_y_start)/spacing)]
        # Determine the ambient flow at each of those points and attach them
        # together
        ambient_flow_at_points = {}
        for i in solution:
            for j in range(len(coordinates)):
                flow_vel = np.sqrt((i['u'][0](coordinates[j]))**2 +
                           (i['u'][1](coordinates[j])**2))
                ambient_flow_at_points.update({flow_vel: coordinates[j]})
        # Sort the coordinates by ambient velocity and extract the best however
        # many we need
        best_velocities = sorted(ambient_flow_at_points.keys())[-num:]
        # Add the turbines at those locations
        for i in best_velocities:
            self.add_turbine(ambient_flow_at_points[i]) 


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
    
