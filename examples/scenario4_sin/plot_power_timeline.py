
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy

f = open("power_timeline.txt", "r")

funcs_dupl = [float(line.strip()) for line in f]
func = []
for i in funcs_dupl:
    if i not in func:
        func.append(i)

it = [float(i)/50*12 for i in range(len(func))]
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
#plt.yticks(numpy.arange(14, basin_x_total/1000, 2))
plt.ylabel(r"Power [MW]")
plt.xlabel(r"Time [h]")
plt.savefig(filename)
plt.close()
