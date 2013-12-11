from dolfin import *

def DistanceToCoast(config):
  V = FunctionSpace(config.domain.mesh, 'CG', 1)
  v = TestFunction(V)
  u = TrialFunction(V)
  f = Constant(1.0)
  y = Function(V)

  bc = DirichletBC(V, 0., config.domain.boundaries, 1)

  print "Solving equations for distance to boundary:"
  print "Solve linear problem for initial guess:"
  # Initialization problem to get good initial guess for nonlinear problem:
  F1 = inner(grad(u), grad(v))*dx - f*v*dx
  solve(lhs(F1)==rhs(F1), y, bc)

  print "Solve nonlinear problem for distance to boundary"
  # Stabilized Eikonal equation 
  eps = Constant(config.domain.mesh.hmax()/25)
  F = sqrt(inner(grad(y), grad(y)))*v*dx - f*v*dx + eps*inner(grad(y), grad(v))*dx
  solve(F==0, y, bc)
  return y
