#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import numpy as np
step = 0.04
maxval = 1.0
fig = plt.figure()
fig.subplots_adjust(bottom=0.5, top=0.75)
ax = Axes3D(fig)

# create supporting points in polar coordinates
r = np.linspace(0, 20, 50)
p = np.linspace(0, 20, 50)
X,Y = np.meshgrid(r,p)

Z1 = 1. - ((X-10)/10)**2
Z1 = np.maximum(Z1, 1e-12)
Z1 = np.exp(-1./Z1)

Z2 = 1. - ((Y-10)/10)**2
Z2 = np.maximum(Z2, 1e-12)
Z2 = np.exp(-1./Z2)
Z = Z1 * Z2 * np.exp(2)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0)
plt.setp(ax.get_zticklabels(), fontsize=20)
plt.setp(ax.get_xticklabels(), fontsize=20)  
plt.setp(ax.get_yticklabels(), fontsize=20)

ax.set_xmargin(0)
ax.set_ymargin(0)
#ax.grid(b=None)
rc('text', usetex = True)
scaling = 1.7
plt.figure(1, figsize = (scaling*7., scaling*4.))

plt.savefig('turbine_plot.pdf', facecolor='white',bbox_inches='tight')
