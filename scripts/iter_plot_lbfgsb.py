#!/usr/bin/python
import sys
import re
import matplotlib.pyplot as plt
from matplotlib import rc
filename = sys.argv[1]
f = open(filename, "r")

# Read the output file of the optimisation run and parse it
# This script assumes that the scipy.L-BFGS-B algorithm was used.

# The iteration numbers
it = []
last_it = 0
# The functional values
func = []
finished = False
# Rescale the  functional values before plotting
# Note: this does not yet include the automatic scaling performed by the turbine code. We will recover that factor from the output file
rescale = 10**-6
for line in f:
    if "Optimization terminated successfully" in line:
        finished = True
    elif "The automatic scaling factor was set to" in line:
        m = re.match(r".*to ([0-9|\-]+\.[0-9|e|E|-]+)", line) 
        print "Found rescaling factor: %s"% m.group(1)
        rescale /= float(m.group(1))

    elif "At iterate" in line:
        it.append(float(line.split()[2]))
        func.append(float(line.split()[4].replace('D', 'e'))*rescale)

f.close()

print "Power output of initial layout: ", func[0]
print "Power output of initial layout: ", func[-1]
print "Relative power increase: ", func[-1]/func[0]

# The produce a nice plot
filename = "iter_plot.pdf"
scaling = 0.5
rc('text', usetex = True)
plt.figure(1, figsize = (scaling*7., scaling*4.))
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.plot(it, func, color = 'black')
#plt.yscale('log')
#plt.axis([0, times[-1], -2.5, 2.5])
#plt.xticks(numpy.arange(0, times[-1]+1, 5))
#plt.yticks(numpy.arange(14, basin_x_total/1000, 2))
plt.ylabel(r"Average power [MW]")
plt.xlabel(r"Optimisation iteration")
plt.savefig(filename)
plt.close()
