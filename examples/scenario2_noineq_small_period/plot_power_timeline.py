
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy

f = open("power_timeline.txt", "r")

funcs_dupl = [float(line.strip()) for line in f]
func = []
for i in funcs_dupl:
    if i not in func:
        func.append(i)

period = 10
it = [float(i)/len(func)*period for i in range(len(func))]
func = numpy.array(func)*1e-6

filename = "power_timeline.pdf"
scaling = 0.5
rc('text', usetex = True)
plt.figure(1, figsize = (scaling*7., scaling*4.))
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.plot(it, func, color = 'black')
#plt.yscale('log')
#plt.axis([0, times[-1], -2.5, 2.5])
#plt.xticks(numpy.arange(0, times[-1]+1, 5))
plt.yticks(numpy.arange(0, 50, 10))
plt.ylabel(r"Power [MW]")
plt.xlabel(r"Time [min]")
plt.savefig(filename)
plt.close()
