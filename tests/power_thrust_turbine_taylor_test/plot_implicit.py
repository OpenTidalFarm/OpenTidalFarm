from dolfin import *
import ufl

mesh = UnitIntervalMesh(100)
V = FunctionSpace(mesh, "CG", 1)
Vdg = FunctionSpace(mesh, "DG", 0)

u0 = project(Expression("1+sin(2*pi*x[0])"), V)
plot(u0, interactive=True)

u = Function(V)
q = TestFunction(V)

tf = project(Expression("sin(2*pi*x[0])"), V)

def norm_approx(u, alpha=1e-4):
    # A smooth approximation to ||u||
    return sqrt(inner(u, u)+alpha**2)

# Advect 
shift = 0.1
theta = 0.5 
uhalf = Constant(1.-theta)*u0 + Constant(theta)*u
F = (inner(u-u0, q) + Constant(shift)*inner(uhalf.dx(0)*1, q) + Constant(1e-8)*inner(grad(u), grad(q)))*dx 
solve(F == 0, u)

plot(u, interactive=True, title="Computed after shifting")
u_shift = project(Expression("1+sin(2*pi*(x[0]-shift))", shift=shift), V)
plot(u_shift, interactive=True, title="Expected after shifting")

# Diffuse
a = Function(V)
chi = ufl.conditional(ufl.ge(tf, 0.0), 0, 1)
F1 = chi*(inner(a-u0, q) + Constant(1e3)*inner(grad(a), grad(q)))*dx 
invchi = 1-chi
F2 = inner(invchi*a, q)*dx 
F = F1 + F2

solve(F == 0, a)
adg = interpolate(a, Vdg)
f = File("averaged.pvd") 
f << adg
plot(adg, interactive=True, title="Averaged")
