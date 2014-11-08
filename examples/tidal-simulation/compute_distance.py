from dolfin import *
from opentidalfarm import FileDomain
import ufl.algorithms

e = ufl.algorithms.expand_derivatives

domain = FileDomain("../data/meshes/orkney/orkney_utm.xml")
mesh = domain.mesh
facet_ids = domain.facet_ids
dist_ids = [1, 2]

V = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)

v = TestFunction(V)
u = Function(V)
bcs = [DirichletBC(V, 0.0, facet_ids, i) for i in dist_ids]
dr = interpolate(Constant(1.0), R)

# Before we solve the Eikonal equation, let's solve a Laplace equation to
# generate an initial guess
F = inner(grad(u), grad(v))*dx - v*dx
solve(F == 0, u, bcs)

du = Function(V)
ddu = Function(V)

epss = map(lambda x: interpolate(x, R), [Constant(1000), Constant(500)])
for i, eps in enumerate(epss):
  print "Solving Eikonal with eps == ", eps.vector()[0]
  F = inner(sqrt(inner(grad(u), grad(u))), v)*dx - v*dx + eps*inner(grad(u), grad(v))*dx
  solve(F == 0, u, bcs)

  if eps != epss[-1]: # we're not at the last eps
    # first-order continuation algorithm
    G = e(derivative(F, u, du) + derivative(F, eps, dr)) # tangent linearisation
    solve(G == 0, du, homogenize(bcs))
    deps = epss[i+1].vector()[0] - epss[i].vector()[0] # delta_epsilon

    #To do first-order continuation:
    #u.assign(1.0*u + deps*u) # update our guess of u for the next solve
    #continue

    H =  e(derivative(G, u, ddu) + derivative(G, eps, dr)) # tangent quadraticisation
    solve(H == 0, ddu, homogenize(bcs))

    # To do second-order continuation:
    u.assign(1.0*u + deps*du + 0.5*deps*deps*ddu)

dist = File("dist.xml")
dist << u

plot(u, title="Distance to open boundaries")
interactive()
