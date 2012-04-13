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

    values[ 101.5, 33. ] = 640.317213444
    values[ 101.5, 34.5] = 638.095351195
    values[ 101.5, 33.5] = 638.105521744
    values[ 101.5, 34. ] = 644.927618316
    values[ 101.5, 35. ] = 640.317213444

    values[ 102., 34.5] = 645.291716549
    values[ 102., 34.] = 652.143261284
    values[ 102., 33.] = 647.586853698
    values[ 102., 33.5] = 645.300152183
    values[ 102., 35.] = 647.586853698

    try:
        return values[x, y]/min(values.values())*100 - 100 # Relative change with respect to the minimum value
    except:
        print "Returning fake value 0.0 for point ", x, y
        return 0.0

offset_x = 100
offset_y = 33
dx, dy = 0.5, 0.5

x = arange(0., 2.000001, dx)
y = arange(0., 2.000001, dy)
X,Y = meshgrid(x, y)

Z = numpy.vectorize(functional_value)(X + offset_x, Y + offset_y)
# Interpolation: bicupic or nearest
imshow(Z, interpolation='bicubic', cmap=cm.jet, origin='lower', extent=[0, 2, 0, 2])
cbar = colorbar()
cbar.set_label("Relative change in the power output")
axis([0.,2.,0.,2.])

xlabel("Offset x direction to reference position")
ylabel("Offset y direction to reference position")

savefig("functional_convergence.pdf")
