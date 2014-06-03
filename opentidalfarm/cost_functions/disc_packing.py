import pylab
from matplotlib.patches import Circle
from random import gauss, randint
from colorsys import hsv_to_rgb
import math
import random
from matplotlib.path import Path
import matplotlib.patches as patches
import itertools as IT
import copy
import cable_costing as cabcos

#############################################################################
##### IMAGE ->-> POLYGON PATHS ->-> LIST OF POLYGON VERTICES (RETURNED) #####
############# REPLACE WITH ALTERNATE - STRAIGHT FROM OTF OUTPUT #############
#############################################################################

from PIL import Image
import time as time
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import Ward

def analyse_smear(n_clusters):
    if n_clusters < 3:
        print 'Requires at least 3 clusters'
        
    # Generate data
    A = Image.open('./tests/test5.png')
    B = A.convert('L')
    C = np.array(B)
    
    lena = C
    print type(lena)
    #lena = np.array()
    # Downsample the image by a factor of 4
    lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
    X = np.reshape(lena, (-1, 1))
    

    # Define the structure A of the data. Pixels connected to their neighbors.
    connectivity = grid_to_graph(*lena.shape)
    
    
    # Compute clustering
    print("Compute structured hierarchical clustering...")
    st = time.time()
    ward = Ward(n_clusters=n_clusters, connectivity=connectivity).fit(X)
    label = np.reshape(ward.labels_, lena.shape)
    print("Elapsed time: ", time.time() - st)
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)
    
    a = []
    p = []
    v = []
    
    for l in range(n_clusters):
        a.append(pylab.contour(label == l, contours=1, colors=[pylab.cm.spectral(l / float(n_clusters)), ]))
        p.append(a[l].collections[0].get_paths()[0])
        v.append(p[l].vertices)
    del v[0]    
    return v

    #fig = pl.figure()
    #ax = fig.add_subplot(111)
    #patch = []
    #
    #for i in range(len(p)):
    #    patch.append(patches.PathPatch(p[i], facecolor='orange', lw=2))
    #for i in range(len(patch)):
    #    ax.add_patch(patch[i])    
    #
    #ax.set_xlim(-2,100)
    #ax.set_ylim(-2,100)
    #pl.show()
    #print v[0]
    
    
    ## Plot the results on an image
    #pl.figure(figsize=(5, 5))
    #pl.imshow(lena, cmap=pl.cm.gray)
    #print n_clusters
    #for l in range(n_clusters):
    #    pl.contour(label == l, contours=1, colors=[pl.cm.spectral(l / float(n_clusters)), ])
    ##pl.contour(label == 14, contours=1, colors=[pl.cm.spectral(14 / float(n_clusters)), ])
    #pl.xticks(())
    #pl.yticks(())
    #pl.show()
    
#############################################################################
#############################################################################
#############################################################################

#############################################################################
########## PACK TURBINES INTO AREA BOUNDED BY PATH AROUND VERTICES###########
#############################################################################

def plotCircles(circles, paths, site):
    # input is list of circles
    # each circle is a tuple of the form (x, y, r)
    ax = pylab.figure()
    ax1 = ax.add_subplot(111)
    bx = pylab.gca()
    rs = [x[2] for x in circles]
    maxr = max(rs)
    minr = min(rs)
    hue = lambda inc: pow(float(inc - minr)/(1.02*(maxr - minr)), 3)

    patch = []
    for i in range(len(paths)):
        patch.append(patches.PathPatch(paths[i], facecolor='orange', lw=2))
    for i in range(len(patch)):
        ax1.add_patch(patch[i]) 
        
    for circle in circles:
        circ = Circle((circle[0], circle[1]), circle[2])
        color = hsv_to_rgb(hue(circle[2]), 1, 1)
        circ.set_color(color)
        circ.set_edgecolor(color)
        bx.add_patch(circ)
    pylab.axis('scaled')
    #pylab.show()

# pretty tuple indices
X = 0
Y = 1
RADIUS = 2

SORT_PARAM_1 = .80 
SORT_PARAM_2 = .10 
# (1, 0) = totally sorted   - appealing border, very dense center, sparse midradius
# (0, 1), (1, 1) = totally randomized  - well packed center, ragged border

# these constants control how close our points are placed to each other
RADIAL_RESOLUTION = .4
ANGULAR_RESOLUTION = .4

# this keeps the boundaries from touching
PADDING = 0

#def assert_no_intersections(f):
#    def asserter(*args, **kwargs):
#        circles = f(*args, **kwargs)
#        intersections = 0
#        for c1 in circles:
#            for c2 in circles:
#                if c1 is not c2 and distance(c1, c2) < c1[RADIUS] + c2[RADIUS]:
#                    intersections += 1
#                    break
#        print "{0} intersections".format(intersections)
#        if intersections:
#            raise AssertionError('Doh!')
#        return circles
#    return asserter


#@assert_no_intersections
def positionCircles(rn):

    points = base_points(ANGULAR_RESOLUTION, RADIAL_RESOLUTION)
    free_points = []
    radii = fix_radii(rn)

    circles = []
    point_count = 0
    for radius in radii:
        print "{0} free points available, {1} circles placed, {2} points examined".format(len(free_points), len(circles), point_count)
        i, L = 0, len(free_points)
        while i < L:
            if available(circles, free_points[i], radius):
                make_circle(free_points.pop(i), radius, circles, free_points)
                break  
            else:
                i += 1   
        else:
            for point in points:
                point_count += 1
                if available(circles, point, radius):
                    make_circle(point, radius, circles, free_points) 
                    break
                else:
                    if not contained(circles, point):
                        free_points.append(point)
    return circles


def fix_radii(radii):
    radii = sorted(rn, reverse=True)
    radii_len = len(radii)

    section1_index = int(radii_len * SORT_PARAM_1)
    section2_index = int(radii_len * SORT_PARAM_2)

    section1, section2 = radii[:section1_index], radii[section1_index:]
    random.shuffle(section2)
    radii = section1 + section2

    section1, section2 = radii[:section2_index], radii[section2_index:]
    random.shuffle(section1)
    return section1 + section2


def make_circle(point, radius, circles, free_points):
    new_circle = point + (radius, )
    circles.append(new_circle)
    i = len(free_points) - 1
    while i >= 0:
        if contains(new_circle, free_points[i]):
            free_points.pop(i)
        i -= 1
             
             
def available(circles, point, radius):
    for circle in circles:
        if distance(point, circle) < radius + circle[RADIUS] + PADDING:
            return False
    return True
        
        
def base_points(radial_res, angular_res):
    circle_angle = 2 * math.pi
    r = 0
    while 1:
        theta = 0
        while theta <= circle_angle:
            yield (r * math.cos(theta), r * math.sin(theta))
            r_ = math.sqrt(r) if r > 1 else 1
            theta += angular_res/r_
        r += radial_res    


def distance(p0, p1):
    return math.sqrt((p0[X] - p1[X])**2 + (p0[Y] - p1[Y])**2) 

def contains(circle, point):
    return distance(circle, point) < circle[RADIUS] + PADDING

def contained(circles, point):
    return any(contains(c, point) for c in circles)
      
def harvest2(circles, p):
    print circles, p
    safe_list = []
    for i in range(len(p)):
        
        for j in range(len(circles)):
            
            centre = [circles[j][0],circles[j][1]]
            print centre
            if p[i].contains_point(centre) == 1:
                safe_list.append(circles[j])
        for circle in safe_list:
            if circle in circles:
                circles.remove(circle)
    return safe_list

def plot_areas(paths, site, v):
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    patch = []
    for i in range(len(paths)):
        patch.append(patches.PathPatch(paths[i], facecolor='orange', lw=2))
    for i in range(len(patch)):
        ax.add_patch(patch[i])
        ax.text(v[i][0][0], v[i][0][1], '%s' % (str(i)))
    ax.set_xlim(site[0],site[1])
    ax.set_ylim(site[2],site[3])
    print 'NOW!'
    pylab.show()
    
def area_of_polygon(x, y):
    area = 0.0
    for i in xrange(-1, len(x) - 1):
        area += x[i] * (y[i + 1] - y[i - 1])
    return area / 2.0

def centroid_of_polygon(points):
    area = area_of_polygon(*zip(*points))
    result_x = 0
    result_y = 0
    N = len(points)
    points = IT.cycle(points)
    x1, y1 = next(points)
    for i in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= (area * 6.0)
    result_y /= (area * 6.0)
    return (result_x, result_y)



#############################################################################
###### COULD PROBABLY USE HOUSE KEEPING AND DISTILLING INTO FUNCTIONS #######
#############################################################################
    
if __name__ == '__main__':
    minrad, maxrad = (1, 1.0001)
    numCircles = 40
    site_x_start = 4
    site_x_length = 40
    site_y_start = 4
    site_y_length = 40    
    site = [0, 100, 0, 50] 
    circles_for_area =[]
    p = []
        
    v = analyse_smear(3)
    print v, type(v)
    for i in range(len(v)):
        p.append(Path(v[i]))
        
    plot_areas(p, site, v)
    
    for j in range(len(v)):
        cent = centroid_of_polygon(v[j])
        v_temp = copy.deepcopy(v[j])
        for i in range(len(v[j])):
            v_temp[i][0] -= cent[0]
            v_temp[i][1] -= cent[1]
        p_temp = [Path(v_temp)]
                          #overall site dimensions [min_x, max_x, min_y, max_y]
        rn = [((maxrad-minrad)*gauss(0,1) + minrad) for x in range(numCircles)]
        circles = positionCircles(rn)
        circles = harvest2(circles, p_temp)
        for i in range(len(circles)):
            circles_for_area.append([circles[i][0] + cent[0], circles[i][1] + cent[1], circles[i][2]])

    plotCircles(circles_for_area, p, site)
    
    turbine_locations = []
    for i in range(len(circles_for_area)):
        turbine_locations.append([circles_for_area[i][0], circles_for_area[i][1]])
        
    CC = cabcos.CableCostGA(substation_location = [[0,0]], show_prog = True, show_result = True)
    output = CC.compute_cable_cost(turbine_locations, prev_routing = None)
    print output[0], output[1]
    deriv = CC.compute_cable_cost_derivative(turbine_locations)
    print deriv
    
    
    
# Notes: should add a function which determines the radius of each turbine patch and adjusts the number of turbines generated
#            accordingly such that sufficient (and not far too many) turbines are generated before being culled.