#!/usr/bin/env python
from pylab import *
import numpy

def functional_value(x,y):
    values = {}
    values[ 100.,   34.] = 652.147265979
    values[ 100. ,   34.5] = 645.295677838
    values[ 100. ,   33.5] = 645.304113617
    values[ 100.,   33.] = 647.590828646
    values[ 100.,   35.] = 647.590828646
    values[ 100.5,   34. ] = 647.293690008
    values[ 100.5,   33.5] = 640.487184586
    values[ 100.5,   34.5] = 640.480326407
    values[ 100.5,   33. ] = 642.740936114
    values[ 100.5,   35. ] = 642.740936114
    values[ 101. ,   33.5] = 640.59468813
    values[ 101.,   33.] = 642.796161855
    values[ 101.,   34.] = 647.45568795
    values[ 101. ,   34.5] = 640.585912104
    values[ 101.,   35.] = 642.796161855
    return values[x, y]/min(values.values())*100 - 100 # Relative change with respect to the minimum value

# make these smaller to increase the resolution
dx, dy = 0.5, 0.5

x = arange(100, 101.000001, dx)
y = arange(33, 35.000001, dy)
X,Y = meshgrid(x, y)

Z = numpy.vectorize(functional_value)(X, Y)
pcolor(X, Y, Z)
colorbar()
axis([100,102,33,35])

show()
