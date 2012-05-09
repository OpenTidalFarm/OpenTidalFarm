#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
step = 0.04
maxval = 1.0
fig = plt.figure()
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

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
#ax.set_zlim3d(0, 1)
#ax.set_xlabel(r'$\phi_\mathrm{real}$')
#ax.set_ylabel(r'$\phi_\mathrm{im}$')
#ax.set_zlabel(r'$V(\phi)$')
plt.savefig('turbine_plot.pdf', facecolor='gray')
plt.show()
