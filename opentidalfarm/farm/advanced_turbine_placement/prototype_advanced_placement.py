import numpy
import dolfin
import opentidalfarm as otf
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


class AdvancedTurbinePlacementParameters(object):
    """ Parameters for the advanced turbine placement algorithms
    """
    def __init__(self):
        self.number_of_turbines = 15
        self.auto_size_array = True 
        self.sizing_functional = otf.FinancialFunctional()
        self.array_sizing = False
        self.record = []


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
        self.problem = self.solver.problem
        self.problem_params = reduced_functional._problem_params
        self.farm = reduced_functional._problem_params.tidal_farm
        self.placement_parameters = self.farm.turbine_placement_parameters
        self.functional = reduced_functional.functional
        self.record = self.placement_parameters.record

        # Perform some preliminary tasks
        self.construct_grid()


    def construct_grid_superceded(self):
        """ Advanced turbine layout protocols add turbines to the domain based
        on a grid derived from the minimum distance constraints of the turbines.
        This method constructs that grid
        """
        spacing = self.farm.turbine_specification.minimum_distance+2
        site_x_start = self.problem_params.tidal_farm.site_x_start
        site_x_end = self.problem_params.tidal_farm.site_x_end
        site_y_start = self.problem_params.tidal_farm.site_y_start
        site_y_end = self.problem_params.tidal_farm.site_y_end
        # Let's store these somewhere more convenient
        self.site = {'site_x_start':site_x_start, 'site_x_end':site_x_end,
                     'site_y_start':site_y_start, 'site_y_end':site_y_end}
        self.grid = [(x*spacing+site_x_start, y*spacing+site_y_start) for x in
                     range(int((site_x_end-site_x_start)/spacing)) for y in
                     range(int((site_y_end-site_y_start)/spacing))]


    def construct_grid(self):
        """ Advanced turbine layout protocols add turbines to the domain based
        on a grid derived from the minimum distance constraints of the turbines.
        This method constructs that grid
        """
        minimum_distance = self.farm.turbine_specification.minimum_distance + 2
        placed_turbines = self.farm.turbine_positions
        
        #TODO Link spacing to mesh? Extract grid from mesh nodes?
        spacing = 5

        exclusion_zones = []
        for turbine in placed_turbines:
            exclusion = Circle((turbine[0], turbine[1]), minimum_distance)
            exclusion_zones.append(exclusion)

        site_x_start = self.problem_params.tidal_farm.site_x_start
        site_x_end = self.problem_params.tidal_farm.site_x_end
        site_y_start = self.problem_params.tidal_farm.site_y_start
        site_y_end = self.problem_params.tidal_farm.site_y_end
        # Let's store these somewhere more convenient
        self.site = {'site_x_start':site_x_start, 'site_x_end':site_x_end,
                     'site_y_start':site_y_start, 'site_y_end':site_y_end}
        grid = [(x*spacing+site_x_start, y*spacing+site_y_start) for x in
                range(int((site_x_end-site_x_start)/spacing)) for y in
                range(int((site_y_end-site_y_start)/spacing))]

        removers = []
        for point in grid:
            for exclusion_zone in exclusion_zones: 
                if exclusion_zone.contains_point(point):
                    removers.append(point)         
        for point in removers:
            if point in grid:
                grid.remove(point) 
        self.grid = grid


    def find_ambient_flow_field(self):
        """ Solve a dummy of the real problem to yield the ambient flow (i.e.
        the flow over the domain in the abscence of turbines)
        """
        ambient_solve = self.solver.solve()
        ambient_state = ambient_solve.next()
        ambient_state = ambient_solve.next()
        self.ambient_state = ambient_state['state']


        state = self.ambient_state

        mesh = self.problem_params.domain.mesh
        coords = mesh.coordinates()

        Output_V = dolfin.VectorFunctionSpace(mesh, 'CG', 1, dim=2)
        u_out = dolfin.TrialFunction(Output_V)
        v_out = dolfin.TestFunction(Output_V)
        M_out = dolfin.assemble(dolfin.inner(v_out, u_out) * dolfin.dx)

        out_state = dolfin.Function(Output_V)
        rhs = dolfin.assemble(dolfin.inner(v_out, state.split()[0]) * dolfin.dx)
        dolfin.solve(M_out, out_state.vector(), rhs, 'cg', 'sor')

        out_state = out_state.vector().array()
        velocity = out_state.reshape(len(out_state)/2,2).transpose()
        velocity = numpy.sqrt(velocity[0]**2 + velocity[1]**2)

        x = coords[:,0]
        y = coords[:,1]


    def find_ambient_velocity_on_grid(self):
        """ Convert the ambient flow field into a dictionary of velocities at
        their corresponding points on the grid
        """
        ambient_velocity_on_grid = {}
        self.find_ambient_flow_field()
        u = self.ambient_state
        for i in range(len(self.grid)):
            flow_vel = numpy.sqrt((u[0](self.grid[i]))**2 +
                                  (u[1](self.grid[i])**2))
            ambient_velocity_on_grid.update({flow_vel:self.grid[i]})
        self.ambient_velocity_on_grid = ambient_velocity_on_grid


    def find_best_point(self, dictionary):
        """ Find the grid point with the best performance from the dictionary
        and return the cartesian coordinate
        """
        best_performance = sorted(dictionary.keys())[-1]
        best_location = dictionary[best_performance]
        return best_location


    def place_turbine(self, location, turbine_number):
        """ Add the turbines at the defined location
        """
        dolfin.info('Current turbines at: ')
        print self.farm.turbine_positions
        dolfin.info('Placing turbine number %i' % turbine_number) 
        self.farm.add_turbine(location)
        self.farm.update()
        self.construct_grid()

    def plot_turbine_positions(self):
        """ Plots the turbines in the order they were added
        """
        import pylab as plt
        import scipy.interpolate

        state = self.ambient_state 
        mesh = self.problem_params.domain.mesh
        coords = mesh.coordinates()

        Output_V = dolfin.VectorFunctionSpace(mesh, 'CG', 1, dim=2)
        u_out = dolfin.TrialFunction(Output_V)
        v_out = dolfin.TestFunction(Output_V)
        M_out = dolfin.assemble(dolfin.inner(v_out, u_out) * dolfin.dx)

        out_state = dolfin.Function(Output_V)
        rhs = dolfin.assemble(dolfin.inner(v_out, state.split()[0]) * dolfin.dx)
        dolfin.solve(M_out, out_state.vector(), rhs, 'cg', 'sor')

        out_state = out_state.vector().array()
        velocity = out_state.reshape(len(out_state)/2,2).transpose()
        velocity = numpy.sqrt(velocity[0]**2 + velocity[1]**2)
        x = coords[:,0]
        y = coords[:,1]

        # Set up a regular grid of interpolation points
        xi, yi = numpy.linspace(x.min(), x.max(), 300), numpy.linspace(y.min(), y.max(), 300)
        xi, yi = numpy.meshgrid(xi, yi)

        # Interpolate; there's also method='cubic' for 2-D data such as here
        zi = scipy.interpolate.griddata((x, y), velocity, (xi, yi), method='linear')
        plt.imshow(zi, vmin=0, vmax=2.5, origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()])
        # TODO Set colorbar limits somehow 
        plt.colorbar() 

        turb_coordinates = self.farm.turbine_positions
        if type(turb_coordinates) != numpy.ndarray:
            turb_coordinates = numpy.array(turb_coordinates)
        plt.plot(turb_coordinates[:,0], turb_coordinates[:,1], 'o', color='green')
        for i in range(len(turb_coordinates)):
            plt.text(turb_coordinates[i][0]+0.5, turb_coordinates[i][1]+0.5, '%s' % (str(i)))
        grid = numpy.array(self.grid)
        plt.plot(grid[:,0], grid[:,1], ',', color='black')
        plt.plot([self.site['site_x_start'], self.site['site_x_end'],
                  self.site['site_x_end'], self.site['site_x_start'], self.site['site_x_start']],
                 [self.site['site_y_start'], self.site['site_y_start'], 
                  self.site['site_y_end'], self.site['site_y_end'], self.site['site_y_start']], 
                 linestyle='--', color='r')
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
        plt.gca().set_aspect('equal', adjustable='box')
        fig = plt.gcf()
        # TODO scale figsize by domain aspect ratio
        fig.set_size_inches(15, 15)
        fig.savefig('test_01_%s.png' % (str(i)))
        fig.clf()


class Sample(object):
    """ A storage vessel to contain the details of each sample run
    """

    def __init__(self, number_of_turbines, functional, power, fidelity, 
                 turbine_locations):

        self.size = number_of_turbines
        self.functional = functional
        self.power = power
        self.fidelity = fidelity
        self.turbine_locations = turbine_locations
