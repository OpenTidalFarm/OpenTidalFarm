import random
from dolfin import * 
from dolfin_adjoint import convergence_order
from numpy import array, dot

def test_gradient_array(J, dJ, x, seed=0.01, perturbation_direction=None):
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

  dj = dJ(x)
  info("Absolute functional evaluation differences: %s" % str(no_gradient))
  info("Gradient approximated with finite differences: %s" % str([(functional_values[i] - j_direct)/perturbations[i] for i in range(len(no_gradient))]))
  info("Gradient computed using the adjoint solution: %s" % str(dj))
  info("Convergence orders for Taylor remainder without adjoint information (should all be 1): %s" % str(convergence_order(no_gradient)))

  with_gradient = []
  for i in range(len(perturbations)):
      remainder = abs(functional_values[i] - j_direct - dot(perturbations[i], dj))
      with_gradient.append(remainder)

  info("Convergence orders for Taylor remainder with adjoint information (should all be 2): %s" % str(convergence_order(with_gradient)))

  return min(convergence_order(with_gradient))

def save_to_file_scalar(function, basename):
    u_out, p_out = output_files(basename)

    M_p_out, q_out, p_out_func = p_output_projector(function.function_space())

    # Project the solution to P1 for visualisation.
    rhs = assemble(inner(q_out,function)*dx)
    solve(M_p_out, p_out_func.vector(),rhs,"cg","sor", annotate=False) 
    
    p_out << p_out_func

def save_to_file(function, basename):
    u_out,p_out = output_files(basename)

    M_u_out, v_out, u_out_func = u_output_projector(function.function_space())
    M_p_out, q_out, p_out_func = p_output_projector(function.function_space())

    # Project the solution to P1 for visualisation.
    rhs = assemble(inner(v_out,function.split()[0])*dx)
    solve(M_u_out, u_out_func.vector(), rhs, "cg", "sor", annotate=False) 
    
    # Project the solution to P1 for visualisation.
    rhs = assemble(inner(q_out,function.split()[1])*dx)
    solve(M_p_out, p_out_func.vector(), rhs, "cg", "sor", annotate=False) 
    
    u_out << u_out_func
    p_out << p_out_func

def u_output_projector(W):
    # Projection operator for output.
    Output_V=VectorFunctionSpace(W.mesh(), 'CG', 1, dim=2)
    
    u_out=TrialFunction(Output_V)
    v_out=TestFunction(Output_V)
    
    M_out=assemble(inner(v_out,u_out)*dx)
    
    out_state=Function(Output_V)

    return M_out, v_out, out_state

def p_output_projector(W):
    # Projection operator for output.
    Output_V=FunctionSpace(W.mesh(), 'CG', 1)
    
    u_out=TrialFunction(Output_V)
    v_out=TestFunction(Output_V)
    
    M_out=assemble(inner(v_out,u_out)*dx)
    
    out_state=Function(Output_V)

    return M_out, v_out, out_state

def output_files(basename):
        
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

