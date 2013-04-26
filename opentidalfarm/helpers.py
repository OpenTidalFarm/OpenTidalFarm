from __future__ import print_function
import random
from dolfin import * 
from dolfin_adjoint import *
from numpy import array, dot
import pylab 
import dolfin

def info_green(*args, **kwargs):
    if MPI.process_number() == 0:
        dolfin.info_green(*args, **kwargs)

def info_red(*args, **kwargs):
    if MPI.process_number() == 0:
        dolfin.info_red(*args, **kwargs)

def info_blue(*args, **kwargs):
    if MPI.process_number() == 0:
        dolfin.info_blue(*args, **kwargs)

def info(*args, **kwargs):
    if MPI.process_number() == 0:
        dolfin.info(*args, **kwargs)

def print0(*args, **kwargs):
    if MPI.process_number() == 0:
        print(*args, **kwargs)

def test_gradient_array(J, dJ, x, seed = 0.01, perturbation_direction = None, plot_file = None):
  '''Checks the correctness of the derivative dJ.
     x must be an array that specifies at which point in the parameter space
     the gradient is to be checked. The functions J(x) and dJ(x) must return 
     the functional value and the functional derivative respectivaly. 

     This function returns the order of convergence of the Taylor
     series remainder, which should be 2 if the gradient is correct.'''

  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  info("Running Taylor remainder convergence analysis to check the gradient ... ")

  # First run the problem unperturbed
  j_direct = J(x)

  # Randomise the perturbation direction:
  if perturbation_direction is None:
    perturbation_direction = x.copy()
    for i in range(len(x)):
      perturbation_direction[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values = []
  perturbations = []
  perturbation_sizes = [seed/(2**i) for i in range(5)]
  for perturbation_size in perturbation_sizes:
    perturbation = perturbation_direction.copy() * perturbation_size
    perturbations.append(perturbation)

    perturbed_x = x.copy() + perturbation 
    functional_values.append(J(perturbed_x))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_j - j_direct) for perturbed_j in functional_values]

  dj = dJ(x, forget = True)
  info_green("Absolute functional evaluation differences: %s" % str(no_gradient))
  info_green("Convergence orders for Taylor remainder without adjoint information (should all be 1): %s" % str(convergence_order(no_gradient)))

  with_gradient = []
  for i in range(len(perturbations)):
      remainder = abs(functional_values[i] - j_direct - dot(perturbations[i], dj))
      with_gradient.append(remainder)

  info_green("Absolute functional evaluation differences with adjoint: %s" % str(with_gradient))
  info_green("Convergence orders for Taylor remainder with adjoint information (should all be 2): %s" % str(convergence_order(with_gradient)))

  if plot_file:
      first_order = [x for x in perturbation_sizes]
      second_order = [x**2 for x in perturbation_sizes]

      pylab.figure()
      pylab.loglog(perturbation_sizes, first_order, 'b--', perturbation_sizes, second_order, 'g--', perturbation_sizes, no_gradient, 'bo-', perturbation_sizes, with_gradient, 'go-') 
      pylab.legend(('First order convergence', 'Second order convergence', 'Taylor remainder without gradient', 'Taylor remainder with gradient'), 'lower right', shadow=True, fancybox=True)
      pylab.xlabel("Perturbation size")
      pylab.ylabel("Taylor remainder")
      pylab.savefig(plot_file)

  return min(convergence_order(with_gradient))

def save_to_file_scalar(function, basename):
    out_file = File(basename+".pvd", "compressed")
    out_file << function 

class StateWriter:
    def __init__(self, config):
        self.u_out, self.p_out = self.output_files(config.finite_element.func_name)
        self.M_u_out, self.v_out, self.u_out_state = self.u_output_projector(config.function_space)
        self.M_p_out, self.q_out, self.p_out_state = self.p_output_projector(config.function_space)

    def write(self, state):
        rhs = assemble(inner(self.v_out, state.split()[0])*dx)
        solve(self.M_u_out, self.u_out_state.vector(), rhs, "cg", "sor", annotate=False) 
        rhs = assemble(inner(self.q_out, state.split()[1])*dx)
        solve(self.M_p_out, self.p_out_state.vector(), rhs, "cg", "sor", annotate=False)

        self.u_out << self.u_out_state
        self.p_out << self.p_out_state

    def u_output_projector(self, W):
        # Projection operator for output.
        Output_V = VectorFunctionSpace(W.mesh(), 'CG', 1, dim=2)
        
        u_out = TrialFunction(Output_V)
        v_out = TestFunction(Output_V)
        
        M_out = assemble(inner(v_out,u_out)*dx)
        out_state = Function(Output_V)

        return M_out, v_out, out_state

    def p_output_projector(self, W):
        # Projection operator for output.
        Output_V = FunctionSpace(W.mesh(), 'CG', 1)
        
        u_out = TrialFunction(Output_V)
        v_out = TestFunction(Output_V)
        
        M_out = assemble(inner(v_out,u_out)*dx)
        out_state = Function(Output_V)

        return M_out, v_out, out_state

    def output_files(self, basename):
            
        # Output file
        u_out = File(basename+"_u.pvd", "compressed")
        p_out = File(basename+"_p.pvd", "compressed")

        return u_out, p_out

def cpu0only(f):
  ''' A decorator class that only evaluates on the first CPU in a parallel environment. '''
  def decorator(self, *args, **kw):
    myid = MPI.process_number()
    if myid == 0:
      f(self, *args, **kw)

  return decorator

