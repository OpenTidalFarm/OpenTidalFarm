import numpy 
import pylab
from dolfin import MPI

def cpu0only(f):
  ''' A decorator class that only evaluates on the first CPU in a parallel environment. '''
  def decorator(self, *args, **kw):
    myid = MPI.process_number()
    if myid == 0:
      f(self, *args, **kw)

  return decorator

class AnimatedPlot:
  ''' A class that implements animated plots with equidistant x values. New points are added by calling the addPoint function. '''
  @cpu0only 
  def __init__(self, xlabel=None, ylabel=None):
    self.datay = []
    pylab.ion() # animation on
    #Note the comma after line. This is placed here because plot returns a list of lines that are drawn.
    self.line, = pylab.plot([], [])
    if xlabel:
      pylab.xlabel(xlabel)
    if ylabel:
      pylab.ylabel(ylabel)
  
  @cpu0only 
  def addPoint(self, value):
    ''' Adds a new point and updates the plot. '''
    #myid = MPI.process_number()
    #if myid != 0:
    #  return

    datay = self.datay
    line = self.line

    datay.append(value) 
    pylab.axis([0, len(datay), 0, 1.1*max(datay)])
    line.set_xdata(range(len(datay)))  # update the data
    line.set_ydata(datay)
    pylab.draw() # draw the points again

  @cpu0only 
  def savefig(self, name):
    pylab.savefig(name)

if __name__ == '__main__':
  plot = AnimatedPlot(xlabel='X Axis', ylabel='Y Axis')
  for i in range(100):
    r = i * numpy.fix(numpy.random.rand()*3)
    plot.addPoint(r)
  plot.savefig("animated_plot.png")
