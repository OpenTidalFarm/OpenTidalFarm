#!/usr/bin/python
import sys
import re
import matplotlib.pyplot as plt
from matplotlib import rc
filename = sys.argv[1]
f = open(filename, "r")

# Read the output file of the optimisation run and parse it
# This script assumes that scipy.slsqp was used.

# The iteration numbers
it = []
last_it = 0
# The functional values
func = []
# The number of functional evaluations
func_evals = []
finished = False
# Rescale the  functional values before plotting
# Note: this does not yet include the automatic scaling performed by the turbine code. We will recover that factor from the output file
rescale = 10**-6
for line in f:
    if "Optimization terminated successfully" in line:
	finished = True
    elif "The automatic scaling factor was set to" in line:
	m = re.match(r".*to ([0-9|\-]+\.[0-9]+)", line) 
	print "Found rescaling factor: %s"% m.group(1)
	rescale /= float(m.group(1))
    if finished:
	if "Current function value" in line:
	    f_opt = float(re.match(r".*: ([0-9|\-]+\.[0-9]+)", line).group(1))
	    print "Current function value: %f" % (f_opt*rescale)
	else:
	    print line,
    # The sqp iterations output are characteristic as they consists of four numers and do not have any commas in the output
    elif "," not in line:
	m = re.match(r"\s+ ([0-9]+) \s+ ([0-9]+) \s+ ([0-9|\.|E\+\-]+) \s+ ([0-9|\.|E\+\-]+)", line)
        try:
	    if len(it) == 0 or it[-1] != int(m.group(1)):
		it.append(int(m.group(1)))
		func_evals.append(int(m.group(2)))
		last_it += 1
		func.append(float(m.group(3))*rescale)
		if last_it != it[-1]:
		    raise ValueError, "The iteration counter is out of sync??!"
        except Exception as e:
	    pass
f.close()

finish_iteration = -1
for i in range(len(it)):
    print it[i], ": ", func[i],
    if i > 1:
	rel_change = (func[i]-func[i-1])/func[i]
	print "\t(relative change: ", rel_change , ", functional evaluations: ", func_evals[i], ")"
	if rel_change < 1e-6:
	    finish_iteration = i+1
	    break
    else:
	print ""

if finish_iteration > -1:
    print "Optimisation finised succesfully after %s iterations." % finish_iteration
it = it[:finish_iteration]
func = func[:finish_iteration]

print "Power output of initial layout: ", func[0]
print "Power output of initial layout: ", func[-1]
print "Relative power increase: ", func[-1]/func[0]

# The produce a nice plot
filename = "iter_plot.pdf"
scaling = 0.7
rc('text', usetex = True)
plt.figure(1, figsize = (scaling*7., scaling*4.))
plt.gcf().subplots_adjust(bottom=0.15)
plt.plot(it, func, color = 'black')
#plt.yscale('log')
#plt.axis([0, times[-1], -2.5, 2.5])
#plt.xticks(numpy.arange(0, times[-1]+1, 5))
#plt.yticks(numpy.arange(14, basin_x_total/1000, 2))
plt.ylabel(r"$J$, scaled by $10^{-6}$")
plt.xlabel(r"Iteration")
plt.savefig(filename)
plt.close()
