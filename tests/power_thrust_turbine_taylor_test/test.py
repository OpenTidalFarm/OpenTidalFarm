from dolfin import *
import ufl

mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh, "CG", 1)

u0 = project(Expression("1+sin(2*pi*x[0])"), V)
plot(u0, interactive=True)

u = Function(V)
q = TestFunction(V)

tf = project(Expression("sin(2*pi*x[0])"), V)

chi = ufl.conditional(ufl.ge(tf, 0.0), 0, 1)

nu = Constant(1e0)
F1 = chi*(inner(u-u0, q) + nu*Constant(1e+1)*inner(u.dx(0)*u, q) + nu*Constant(1e-8)*nu*inner(grad(u), grad(q)))*dx 
invchi = 1-chi
F2 = inner(invchi*u, q)*dx 
F = F1 + F2

def norm_approx(u, alpha=1e-4):
    # A smooth approximation to ||u||
    return sqrt(inner(u, u)+alpha**2)

F1 = chi*(inner(u-u0, q) + Constant(10e-1)*inner(u0.dx(0)*u0/norm_approx(u0), q) + Constant(1e-5)*inner(grad(u), grad(q)))*dx 
invchi = 1-chi
F2 = inner(invchi*u, q)*dx 
F = F1 + F2

sol = Function(V)
solve(F == 0, u)
plot(u, interactive = True)
