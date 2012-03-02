from __future__ import print_function
import libadjoint
from dolfin import *
from dolfin_adjoint.solving import *

def pprint(*args, **kw):
  ''' A print that only prints on the first CPU in a parallel environment. '''
  myid = MPI.process_number()
  if myid == 0:
    print(*args)

def replay_dolfin(forget=False):
  if "record_all" not in debugging or debugging["record_all"] is not True:
    pprint("Warning: your replay test will be much more effective with debugging['record_all'] = True.")

  for i in range(adjointer.equation_count):
      (fwd_var, output) = adjointer.get_forward_solution(i)

      storage = libadjoint.MemoryStorage(output)
      storage.set_compare(tol=0.0)
      storage.set_overwrite(True)
      adjointer.record_variable(fwd_var, storage)

      if forget:
        adjointer.forget_forward_equation(i)

def convergence_order(errors):
  import math

  orders = [0.0] * (len(errors)-1)
  for i in range(len(errors)-1):
    orders[i] = math.log(errors[i]/errors[i+1], 2)

  return orders

def adjoint_dolfin(functional, forget=True):
  for i in range(adjointer.equation_count)[::-1]:
      (adj_var, output) = adjointer.get_adjoint_solution(i, functional)
      
      storage = libadjoint.MemoryStorage(output)
      adjointer.record_variable(adj_var, storage)

      if forget:
        adjointer.forget_adjoint_equation(i)

  return output.data # return the last adjoint state

def test_initial_condition_adjoint(J, ic, final_adjoint, seed=0.01, perturbation_direction=None):
  '''forward must be a function that takes in the initial condition (ic) as a dolfin.Function
     and returns the functional value by running the forward run:

       func = J(ic)

     final_adjoint is the adjoint associated with the initial condition
     (usually the last adjoint equation solved).

     This function returns the order of convergence of the Taylor
     series remainder, which should be 2 if the adjoint is working
     correctly.'''

  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  pprint("Running Taylor remainder convergence analysis for the adjoint model ... ")
  import random

  # First run the problem unperturbed
  ic_copy = dolfin.Function(ic)
  f_direct = J(ic_copy)

  # Randomise the perturbation direction:
  if perturbation_direction is None:
    perturbation_direction = dolfin.Function(ic.function_space())
    vec = perturbation_direction.vector()
    for i in range(len(vec)):
      vec[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values = []
  perturbations = []
  for perturbation_size in [seed/(2**i) for i in range(5)]:
    perturbation = dolfin.Function(perturbation_direction)
    vec = perturbation.vector()
    vec *= perturbation_size
    perturbations.append(perturbation)

    perturbed_ic = dolfin.Function(ic)
    vec = perturbed_ic.vector()
    vec += perturbation.vector()

    functional_values.append(J(perturbed_ic))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

  pprint("Taylor remainder without adjoint information: ", no_gradient)
  pprint("Convergence orders for Taylor remainder without adjoint information (should all be 1): ", convergence_order(no_gradient))

  adjoint_vector = final_adjoint.vector()

  with_gradient = []
  for i in range(len(perturbations)):
    remainder = abs(functional_values[i] - f_direct - adjoint_vector.inner(perturbations[i].vector()))
    with_gradient.append(remainder)

  pprint("Taylor remainder with adjoint information: ", with_gradient)
  pprint("Convergence orders for Taylor remainder with adjoint information (should all be 2): ", convergence_order(with_gradient))

  return min(convergence_order(with_gradient))

def test_gradient_array(J, dJ, x, seed=0.01, perturbation_direction=None):
  '''Checks the correctness of the derivative dJ.
     x must be an array that specifies at which point in the parameter space
     the gradient is to be checked. The functions J(x) and dJ(x) must return 
     the functional value and the functional derivative respectivaly. 

     This function returns the order of convergence of the Taylor
     series remainder, which should be 2 if the gradient is correct.'''

  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  pprint("Running Taylor remainder convergence analysis to check the gradient ... ")
  import random
  from numpy import array, dot

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
  for perturbation_size in [seed/(2**i) for i in range(5)]:
    perturbation = perturbation_direction.copy() * perturbation_size
    perturbations.append(perturbation)

    perturbed_x = x.copy() + perturbation 
    functional_values.append(J(perturbed_x))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_j - j_direct) for perturbed_j in functional_values]

  #pprint("The estimated derivative using a first order scheme is: ", [(functional_values[i] - j_direct)/(seed/(2**i)) for i in range(5)])n
  pprint("Convergence orders for Taylor remainder without adjoint information (should all be 1): ", convergence_order(no_gradient))

  dj = dJ(x)

  with_gradient = []
  for i in range(len(perturbations)):
    remainder = abs(functional_values[i] - j_direct - dot(perturbations[i], dj))
    with_gradient.append(remainder)

  pprint("Convergence orders for Taylor remainder with adjoint information (should all be 2): ", convergence_order(with_gradient))

  return min(convergence_order(with_gradient))

def tlm_dolfin(parameter, forget=True):
  for i in range(adjointer.equation_count):
      (tlm_var, output) = adjointer.get_tlm_solution(i, parameter)

      storage = libadjoint.MemoryStorage(output)
      storage.set_overwrite(True)
      adjointer.record_variable(tlm_var, storage)

      if forget:
        adjointer.forget_tlm_equation(i)
  return output

def test_initial_condition_tlm(J, dJ, ic, seed=0.01, perturbation_direction=None):
  '''forward must be a function that takes in the initial condition (ic) as a dolfin.Function
     and returns the functional value by running the forward run:

       func = J(ic)

     final_adjoint is the tangent linear variable for the solution on which the functional depends
     (usually the last TLM equation solved).

     dJ must be the derivative of the functional with respect to its argument, evaluated and assembled at
     the unperturbed solution (a dolfin Vector).

     This function returns the order of convergence of the Taylor
     series remainder, which should be 2 if the adjoint is working
     correctly.'''

  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  pprint("Running Taylor remainder convergence analysis for the tangent linear model... ")
  import random

  # First run the problem unperturbed
  ic_copy = dolfin.Function(ic)
  f_direct = J(ic_copy)

  # Randomise the perturbation direction:
  if perturbation_direction is None:
    perturbation_direction = dolfin.Function(ic.function_space())
    vec = perturbation_direction.vector()
    for i in range(len(vec)):
      vec[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values = []
  perturbations = []
  for perturbation_size in [seed/(2**i) for i in range(5)]:
    perturbation = dolfin.Function(perturbation_direction)
    vec = perturbation.vector()
    vec *= perturbation_size
    perturbations.append(perturbation)

    perturbed_ic = dolfin.Function(ic)
    vec = perturbed_ic.vector()
    vec += perturbation.vector()

    functional_values.append(J(perturbed_ic))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

  pprint("Taylor remainder without tangent linear information: ", no_gradient)
  pprint("Convergence orders for Taylor remainder without tangent linear information (should all be 1): ", convergence_order(no_gradient))

  with_gradient = []
  for i in range(len(perturbations)):
    param = InitialConditionParameter(ic, perturbations[i])
    final_tlm = tlm_dolfin(param).data
    remainder = abs(functional_values[i] - f_direct - final_tlm.vector().inner(dJ))
    with_gradient.append(remainder)

  pprint("Taylor remainder with tangent linear information: ", with_gradient)
  pprint("Convergence orders for Taylor remainder with tangent linear information (should all be 2): ", convergence_order(with_gradient))

  return min(convergence_order(with_gradient))
