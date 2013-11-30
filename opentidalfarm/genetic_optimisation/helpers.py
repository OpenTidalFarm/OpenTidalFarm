from optimiser import GeneticOptimisation
import numpy

def maximize_genetic(optimisation_problem):
    """
    Takes a GeneticOptimisation instance, runs the problem and returns the
    turbine positions (as tuples) for the best solution
    """
    # check if optimisation_problem is a GeneticOptimisation instance
    if not isinstance(optimisation_problem, GeneticOptimisation):
        raise RuntimeError("Must be a GeneticOptimisation instance")
    else:
        optimisation_problem.run()
        final_turbines = optimisation_problem.get_turbine_pos()
        turbines = []
        for i in range(len(final_turbines)/2):
            turbines.append((final_turbines[2*i], final_turbines[2*i+1]))
        return turbines


class PositionGenerator(object):
    """
    Generates a number of turbine layouts to be used as some initial guesses for
    some of the population in a genetic solve.
    """
    def __init__(self, config, number_of_turbines):
        self.tx = config.params["turbine_x"]
        self.ty = config.params["turbine_y"]
        self.minx = config.domain.site_x_start + self.tx*0.5
        self.miny = config.domain.site_y_start + self.ty*0.5
        self.maxx = config.domain.site_x_end - self.tx*0.5
        self.maxy = config.domain.site_y_end - self.ty*0.5
        self.x_range = self.maxx - self.minx
        self.y_range = self.maxy - self.miny
        self.number_of_turbines = number_of_turbines 


    def _along_x(self, y_coordinate):
        """
        Turbines aligned along the x-axis with a constant y coordinate
        """
        try:
            dx = self.x_range/(self.number_of_turbines - 1.)
        except ZeroDivisionError:
            dx = self.x_range/2.
        turbines = [(self.minx+i*dx, y_coordinate) 
                    for i in range(self.number_of_turbines)]
        return turbines


    def _along_y(self, x_coordinate):
        """
        Turbines aligned along the y-axis with a constant x coordinate
        """
        try:
            dy = self.y_range/(self.number_of_turbines - 1.)
        except ZeroDivisionError:
            dy = self.y_range/2.
        turbines = [(x_coordinate, self.miny+i*dy)
                    for i in range(self.number_of_turbines)]
        return turbines


    def along_x_lower(self):
        """
        Returns the turbines along the x axis with a y coordinate at the lowest
        part of the domain
        """
        return self._along_x(self.miny)

    
    def along_x_centre(self):
        """
        Returns the turbines along the x axis with a y coordinate in the centre
        of the domain
        """
        return self._along_x((self.maxy+self.miny)/2.)
    
    
    def along_x_upper(self):
        """
        Returns the turbines along the x axis with a y coordinate at the highest
        part of the domain
        """
        return self._along_x(self.maxy)


    def along_y_left(self):
        """
        Returns the turbines along the y axis with a x coordinate at the
        leftmost point
        """
        return self._along_y(self.minx)
    

    def along_y_centre(self):
        """
        Returns the turbines along the y axis with a x coordinate in the centre
        of the domain
        """
        return self._along_y((self.maxx+self.minx)/2.)
    

    def along_y_right(self):
        """
        Returns the turbines along the y axis with a x coordinate at the
        rightmost point
        """
        return self._along_y(self.maxx)
    
    
    def _circular(self, centre, radius, n):
        """
        Arranges n turbines in a circle around centre with radius r
        """
        dtheta = 2*numpy.pi/n
        turbines = [(centre[0]+radius*numpy.cos(i*dtheta),
                     centre[1]+radius*numpy.sin(i*dtheta)) for i in range(n)]
        return turbines

    
    def one_circle(self):
        """
        Arranges all turbines in a circle
        """
        centre = (((self.maxx+self.minx)/2.), ((self.maxy+self.miny)/2.))
        radius = min([self.x_range*0.5, self.y_range*0.5])
        return self._circular(centre, radius, self.number_of_turbines)


    def two_circles_x(self):
        """
        Arranges turbines in two circles in the x-direction
        """
        c1 = (self.minx+0.25*self.x_range-0.5*self.tx, (self.maxy+self.miny)*0.5)
        c2 = (self.minx+0.75*self.x_range+0.5*self.tx, (self.maxy+self.miny)*0.5)
        radius = min([(self.x_range-self.tx)*0.25, self.y_range*0.5])
        c1_n = self.number_of_turbines/2
        c2_n = self.number_of_turbines - c1_n
        c1 = self._circular(c1, radius, c1_n)
        c2 = self._circular(c2, radius, c2_n)
        return numpy.concatenate((c1, c2))


    def two_circles_y(self):
        """
        Arranges turbines in two circles in the y-direction
        """
        c1 = ((self.maxx+self.minx)*0.5, self.miny+0.25*self.y_range-0.5*self.ty)
        c2 = ((self.maxx+self.minx)*0.5, self.miny+0.75*self.y_range+0.5*self.ty)
        radius = min([(self.y_range-self.ty)*0.25, self.x_range*0.5])
        c1_n = self.number_of_turbines/2
        c2_n = self.number_of_turbines - c1_n
        c1 = self._circular(c1, radius, c1_n)
        c2 = self._circular(c2, radius, c2_n)
        return numpy.concatenate((c1, c2))


    def _normal_deploy(self, nx, ny):
        """
        Arrange nx x ny turbines spread across the domain
        """
        try:
            dx = self.x_range/(nx - 1.)
        except ZeroDivisionError:
            dx = 0.
        try:
            dy = self.y_range/(ny - 1.)
        except ZeroDivisionError:
            dy = 0.
        turbines = []
        for i in range(nx):
            for j in range(ny):
                turbines.append((self.minx+i*dx, self.miny+j*dy))
        return turbines


    def _factors(self, n):
        """
        Returns the factors of n
        """
        factors = set(reduce(list.__add__,
                       ([i,n//i] for i in range(1, int(n**0.5) + 1) if n%i==0)))
        return list(factors)

    def get_all_normal_deploy(self):
        """
        Get a list of turbine positions for all possibilities using the normal
        deploy method
        """
        factors = self._factors(self.number_of_turbines)
        seeds = []
        for i in range(len(factors)):
            seeds.append(self._normal_deploy(factors[i],
                                            self.number_of_turbines/factors[i]))
        return seeds


    def get_all(self):
        """
        Return a list of all possible turbine positions
        """
        positions = self.get_all_normal_deploy()
        positions.append(self.along_x_lower())
        positions.append(self.along_x_centre())
        positions.append(self.along_x_upper())
        positions.append(self.along_y_left())
        positions.append(self.along_y_centre())
        positions.append(self.along_y_right())
        positions.append(self.one_circle())
        positions.append(self.two_circles_x())
        positions.append(self.two_circles_y())
        return positions
