from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import pylab as plb


class HCP_Discretisation(object):
    '''
    a class which packs disks in a hexagonal grid into turbine areas
    
    inputs: radius, vertices of turbine area, site dimensions
        radius - (r) - half the minimum distance allowable between turbines (scalar)
        vertices - (v) - a list of arrays of lists of the vertex points describing where 
            turbines should be placed [np.array([[x1,y1],[x2,y2]...]), np.array([...]), ...]
        site dimesions - (site) list of points describing the overall turbine site
        plot - (plot) True or False to plot output
        
    outputs: list of lists of turbine location and radius - [[x_co-ord, y_co-ord, radius], ...]
    '''

    def __init__(self, r, v, site, plot = False):
        
        self.r = r
        self.v = v
        self.site = site
        self.plot = plot

        p = []
        for i in range(len(self.v)):
            p.append(Path(self.v[i]))
        self.p = p
    
    
    def plot_turbines(self, turbines):
        '''
        input: turbines, turbine area and site dimensions:
            turbines - (turbines) list of lists - [[x_co-ord, y_co-ord, radius], ...]
            turbine area - (paths) list of matplotlib paths around turbine areas - [path0, ...]
            site dimensions - (site) list of site corner co-ords - [min_x, max_x, min_y, max_y]
        
        output: produces plot showing turbine layout
        '''
        #initialise plot
        ax = plb.figure()
        ax0 = plb.gca()
        ax1 = ax.add_subplot(111)
        
        
        #plot turbines
        for turbine in turbines:
            turb = Circle((turbine[0], turbine[1]), turbine[2])
            turb.set_color((float(np.random.rand(1,1)), 0.75, .5))
            turb.set_edgecolor('black')
            ax0.add_patch(turb)
        #plot turbine areas
        patch = []
        for i in range(len(self.p)):
            patch.append(patches.PathPatch(self.p[i], facecolor='none', lw=2))
        for i in range(len(patch)):
            ax1.add_patch(patch[i]) 
        #plot    
        plb.axis('scaled')
        plb.show()
    
    
    def grid_points(self, n, path):
        '''
        input: radius, number to place:
            radius - (radius) a scalar giving the minimum distance radius of the turbines
                can be replaced with x_res, y_res for other shapes later
            number - (n) number of free points to scope
        
        output: list of x, y coordinates in lists [[x_co-ord, y_co-ord], ...]
        '''
        p = Path.get_extents(path).bounds
        x_res, y_res = (np.sqrt(6) / 3) * 2 * self.r, 2 * self.r
        x = np.linspace(p[0], (p[0] + n * x_res), n+1)
        y = np.linspace(p[1], (p[1] + n * y_res), n+1)
        points = []
        for i in range(0,len(x),2):
            for j in range(len(y)):
               points.append([x[i], y[j], self.r]) 
        for i in range(1,len(x),2):
            for j in range(len(y)):
               points.append([x[i], y[j] + (y_res / 2), self.r])  
        self.plot_turbines(points)
        return points
        
    
    def size_array(self, path):
        '''
        input: turbine area and turbine radius
            turbine area - (path) a single matplotlib path
            radius - (r) an scalar
            
        process: determines a sensible number of turbines to generate
        
        output: a scalar representing a sensible grid size for population of turbs
        '''
        p = Path.get_extents(path).bounds
        width = p[2]
        height = p[3]
        if width >= height:
            return (width / (2 * self.r)) + 2
        else: return (height / (2 * self.r)) + 2
    
    
    def harvest(self, path, turbines):
        '''
        input: turbines and turbine areas
            turbines - (turbines) list of lists - [[x_co-ord, y_co-ord, radius], ...]
            turbine area - (paths) list of matplotlib paths around turbine areas - [path0, ...]
        
        process: removes turbines outside turbine areas 
        
        output: list of lists of coords of remaining turbines [[x_co-ord, y_co-ord, radius], ...]
        '''
        safe_list = []
        for i in range(len(path)):
            for j in range(len(turbines)):
                centre = [turbines[j][0],turbines[j][1]]
                if path.contains_point(centre) == 1:
                    safe_list.append(turbines[j])
            for turbine in safe_list:
                if turbine in turbines:
                    turbines.remove(turbine)
        return safe_list
    
    def hcp(self):
        '''
        runs the class
        '''
        turbs = []
        for i in range(len(self.p)):
            n = self.size_array(self.p[i])
            turbines = self.grid_points(n, self.p[i])
            turbs.append(self.harvest(self.p[i], turbines))
        if self.plot:
            turbines = []
            for i in range(len(turbs)):
                for j in range(len(turbs[i])):
                    turbines.append(turbs[i][j])
            self.plot_turbines(turbines)
        return turbines

            
class Useful_Functions(object):
    
    def __init__(self, turbines):
        self.turbines = turbines
        
    def turbine_format(self):
        turbine_locations = []
        for i in range(len(self.turbines)):
            turbine_locations.append([self.turbines[i][0], self.turbines[i][1]])
        return turbine_locations
       
   
if __name__ == '__main__':

    r = 4
    
    v = [np.array([[0,0],[100,0],[100,100],[50,50],[0,100],[0,0]]), np.array([[100,100],[200,100],[200,200],[150,150],[100,200],[100,100]])]
    
    site = []
    
    Culley = HCP_Discretisation(r, v, site, plot = True)  
    
    turbines = Culley.hcp()
    
    Convert = Useful_Functions(turbines)
    
    m = Convert.turbine_format()




















############################################################################
#############################################################################
#############################################################################




def position_turbines(radii, n):
    '''
    input: radii of the turbines, number to place:
        radii - (radii) list of radii for each turbine (length gives n turbines) - [r0, ...]
        number - (n) number of free points to scope
        
    output: list of turbine locations
        format - list of lists - [[x_co-ord, y_co-ord, radius], ...]
    '''
    turbines = []
    point_count = 0
    free_points = grid_points(max(radii), n)
    for radius in radii:
        i = 0 
        L = len(free_points)
        while i < L:
            if not occupied(turbines, free_points[i], radius):
                make_circle(free_points.pop(i), radius, turbines, free_points)
                break  
            else:
                i += 1  
    return turbines
    
def occupied(turbines, coordinate, radius):
    for turbine in turbines:
        if distance(coordinate, turbine) < radius + turbine[2]:
            return True
    return False
    
def distance(point_a, point_b):
    return (np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2))
    
    
def available(circles, point, radius):
    for circle in circles:
        if distance(point, circle) < radius:
            return False
    return True


def base_points(x_res, y_res, n):
    x = []
    for i in range(0, n, 2): x.append(0); x.append(0.5)    
    for i in range(len(x)):
        x[i] = [i*x_res, x[i]*y_res]
    print x

def grid_points(r, n):
    '''
    input: radius, number to place:
        radius - (radius) a scalar giving the minimum distance radius of the turbines
            can be replaced with x_res, y_res for other shapes later
        number - (n) number of free points to scope
    
    output: list of x, y coordinates in lists [[x_co-ord, y_co-ord], ...]
    '''
    x_res, y_res = (np.sqrt(6) / 3) * 2 * r, 2 * r
    x = np.linspace(0, (n * x_res), n+1)
    y = np.linspace(0, (n * y_res), n+1)
    points = []
    for i in range(0,len(x),2):
        for j in range(len(y)):
           points.append([x[i], y[j],r]) 
    for i in range(1,len(x),2):
        for j in range(len(y)):
           points.append([x[i], y[j] + (y_res / 2),r])       
    return points
    
def make_circle(point, radius, circles, free_points):
    new_circle = point + [radius, ]
    circles.append(new_circle)
#    i = len(free_points) - 1
#    while i >= 0:
#        if contains(new_circle, free_points[i]):
#            free_points.pop(i)
#        i -= 1    

    
    
   
    
    


