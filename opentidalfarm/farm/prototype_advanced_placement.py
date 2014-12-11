import numpy
import dolfin
import opentidalfarm as otf

class PrototypeAdvancedTurbinePlacement(object):
    """ Prototypical advanced turbine placement object
    """

    def __init__(self, reduced_functional):
        """ Constructor of the prototypical class
        """
        # Check the reduced_functional is what we want it to be
        if not isinstance(reduced_functional,
                otf.reduced_functional_prototype.ReducedFunctionalPrototype):
            raise TypeError, "reduced_functional must be of type ReducedFunctional"

        # Fish out some useful objects for ease of accessibility
        self.solver = reduced_functional.solver
        self.problem = reduced_functional.problem
        self.problem_params = reduced_functional._problem_params
        self.farm = reduced_functional._problem_params.tidal_farm
        self.placement_parameters = self.farm.turbine_placement_parameters.

        # Perform some preliminary tasks
        self.construct_grid()


    def construct_grid(self):
        """ Advanced turbine layout protocols add turbines to the domain based
        on a grid derived from the minimum distance constraints of the turbines.
        This method constructs that grid
        """
        site_x_start = self.problem_params.tidal_farm.site_x_start
        site_x_end = self.problem_params.tidal_farm.site_x_end
        site_y_start = self.problem_params.tidal_farm.site_y_start
        site_y_end = self.problem_params.tidal_farm.site_y_end
        self.grid = [(x*spacing+site_x_start, y*spacing+site_y_start) for x in
                     range(int((site_x_end-site_x_start)/spacing)) for y in
                     range(int((site_y_end-site_y_start)/spacing))]


    def find_ambient_flow_field(self):
        """ Solve a dummy of the real problem to yield the ambient flow (i.e.
        the flow over the domain in the abscence of turbines)
        """
        ambient_solve = self.solver.solve()
        ambient_state = ambient_solve.next()
        ambient_state = ambient_solve.next()
        self.ambient_state = ambient_state['state']


    def find_ambient_velocity_on_grid(self):
        """ Convert the ambient flow field into a dictionary of velocities at
        their corresponding points on the grid
        """
        ambient_velocity_on_grid = {}
        find_ambient_flow_field()
        u = self.ambient_state
        for i in range(len(coordinates)):
            flow_vel = np.sqrt((u[0](self.grid[i]))**2 +
                               (u[1](self.grid[i])**2))
            ambient_velocity_on_grid.update({flow_vel:self.grid[i]})
        self.ambient_velocity_on_grid = ambient_velocity_on_grid


    def find_best_point(dictionary):
        """ Find the grid point with the best performance from the dictionary
        and return the cartesian coordinate
        """
        best_performance = sorted(dictionary.keys())[-1]
        best_location = dictionary[best_performance]
        return best_location


    def place_turbine(self, location):
        """ Add the turbines at the defined location
        """
        dolfin.info('Current turbines at: ')
        print self.farm.turbine_positions
        dolfin.info('Placing turbine number %i' % i) 
        self.farm.add_turbine(best_location)
        self.farm.update()
