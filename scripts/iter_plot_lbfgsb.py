#!/usr/bin/python
import sys
import re
import matplotlib.pyplot as plt
import argparse
from matplotlib import rc

parser = argparse.ArgumentParser(description='Create a convergence plot from OpenTidalFarm stdout file.')
parser.add_argument('file', type=str, help='the filename containing the OpenTidalFarm stdout')
parser.add_argument('--scale', default=1., type=float, help='rescaling factor (in addition to the MW conversion, i.e. division by 1e-6).')
parser.add_argument('--output', default='iter_plot.pdf', type=str, help='output filename')

args = parser.parse_args()

f = open(args.file, "r")

# Read the output file of the optimisation run and parse it
# This script assumes that the scipy.L-BFGS-B algorithm was used.

found_lbfgsb = False
# The iteration numbers
it = []
last_it = 0
# The functional values
func = []
finished = False
# Rescale the  functional values before plotting
# Note: this does not yet include the automatic scaling performed by the turbine code. We will recover that factor from the output file
rescale = 10**-6 * args.scale
for line in f:
    if "RUNNING THE L-BFGS-B CODE" in line:
        found_lbfgsb = True

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

if not found_lbfgsb:
    print "No L-BFGS-B output found. Please supply the stdout record of an OpenTidalFarm simulation which used the L-BFGS-B optimisation algorithm."
    sys.exit(1)

print "Power output of initial layout: ", func[0]
print "Power output of final layout: ", func[-1]
try:
    print "Relative power increase: ", func[-1]/func[0]
except ZeroDivisionError:
    print "Relative power increase: oo"
print "Number of iterations: ", len(func)

# Produce plot
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
plt.ylabel(r"Power production [MW]")
plt.xlabel(r"Optimisation iteration")
plt.savefig(args.output)
plt.close()
