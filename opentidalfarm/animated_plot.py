import numpy
import pylab
from helpers import cpu0only


class AnimatedPlot:
    ''' A class that implements animated plots with equidistant x values.
    New points are added by calling the addPoint function. '''

    @cpu0only
    def __init__(self, xlabel=None, ylabel=None):
        self.datay = []
        pylab.ion()  # animation on
        # Note the comma after line. This is placed here because plot returns
        # a list of lines that are drawn.
        self.line, = pylab.plot([], [])
        if xlabel:
            pylab.xlabel(xlabel)
        if ylabel:
            pylab.ylabel(ylabel)

    @cpu0only
    def addPoint(self, value):
        ''' Adds a new point and updates the plot. '''
        datay = self.datay
        line = self.line

        datay.append(value)
        pylab.axis([0, len(datay), min(0, 1.1 * min(datay)),
                    max(0, 1.1 * max(datay))])
        line.set_xdata(range(len(datay)))  # update the data
        line.set_ydata(datay)
        pylab.draw()  # draw the points again

    @cpu0only
    def savefig(self, name):
        pylab.savefig(name)

if __name__ == '__main__':
    plot = AnimatedPlot(xlabel='X Axis', ylabel='Y Axis')
    for i in range(100):
        r = i * numpy.fix(numpy.random.rand() * 3)
        plot.addPoint(r)
    plot.savefig("animated_plot.png")
