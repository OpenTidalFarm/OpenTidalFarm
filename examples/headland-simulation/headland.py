#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _channel_simulation:
#
# .. py:currentmodule:: opentidalfarm
#
# Sinusoidal wave in a channel
# ============================
#
#
# Introduction
# ************
#
# This example simulates the flow in a channel with oscillating velocity
# in-/outflow on the west boundary, fixed surface height on the east boundary,
# and free-slip flow on the north and south boundaries.
#
# The shallow water equations to be solved are
#
# .. math::
#       \frac{\partial u}{\partial t} +  u \cdot \nabla  u - \nu \nabla^2 u  + g \nabla \eta + \frac{c_b}{H} \| u \|  u = 0, \\
#       \frac{\partial \eta}{\partial t} + \nabla \cdot \left(H u \right) = 0, \\
#
# where
#
# - :math:`u` is the velocity,
# - :math:`\eta` is the free-surface displacement,
# - :math:`H=\eta + h` is the total water depth where :math:`h` is the
#   water depth at rest,
# - :math:`c_b` is the (quadratic) natural bottom friction coefficient,
# - :math:`\nu` is the viscosity coefficient,
# - :math:`g` is the gravitational constant.
#
# The boundary conditions are:
#
# .. math::
#       u = \begin{pmatrix}\sin(\frac{\pi t}{5}) \frac{y (50-y)}{625}\\0\end{pmatrix} & \quad \textrm{on } \Gamma_1, \\
#       \eta = 0 & \quad \textrm{on } \Gamma_2, \\
#       u \cdot n = 0 & \quad \textrm{on } \Gamma_3, \\
#
# where :math:`n` is the normal vector pointing outside the domain,
# :math:`\Gamma_1` is the west boundary of the channel, :math:`\Gamma_2` is the
# east boundary of the channel, and :math:`\Gamma_3` is the north and south
# boundaries of the channel.
#
#

# After a few timesteps the solution should looks like this:

# .. image:: simulation_result.png
#     :scale: 40
#     :align: center

# Implementation
# **************
#

# We begin with importing the OpenTidalFarm module.

from opentidalfarm import *

# Next we get the default parameters of a shallow water problem and configure it
# to our needs.

prob_params = SWProblem.default_parameters()

# First we define the computational domain. For a simple channel, we can use the
# :class:`RectangularDomain` class:

domain = FileDomain("mesh/headland.xml")

# The boundary of the domain is marked with integers in order to specify
# different boundary conditions on different parts of the domain. You can plot
# and inspect the boundary ids with:

#plot(domain.facet_ids)
#interactive()

# Once the domain is created we attach it to the problem parameters:

prob_params.domain = domain

# Next we specify boundary conditions. For time-dependent boundary condition use
# a parameter named `t` in the :class:`dolfin.Expression` and it will be automatically be
# updated to the current timelevel during the solve.

# We assume a propagating wave in the headland case and compute the phase
# difference to drive flow in the channel.

tidal_amplitude = 2.
tidal_period = 12.42*60*60 # M2 tidal period
H = 80 # channel depth

bcs = BoundaryConditionSet()
eta_channel = "amp*sin(omega*t + omega/pow(g*H, 0.5)*x[0])"
eta_expr = Expression(eta_channel, t=Constant(0), amp=tidal_amplitude,
                      omega=2*pi/tidal_period, g=9.81, H=H)
bcs.add_bc("eta", eta_expr, facet_id=1, bctype="strong_dirichlet")
bcs.add_bc("eta", eta_expr, facet_id=2, bctype="strong_dirichlet")

# No-slip boundary conditions on the headland sides are enforced by using
# `strong_dirichlet` boundary conditions.

bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="strong_dirichlet")

# Again we attach boundary conditions to the problem parameters:

prob_params.bcs = bcs

# The other parameters are straight forward:

# Equation settings
nu = 20
prob_params.viscosity = Constant(nu)
prob_params.depth = Constant(H)
prob_params.friction = Constant(0.0025)
# Temporal settings
prob_params.theta = Constant(0.5)
prob_params.start_time = Constant(0)
prob_params.finish_time = Constant(tidal_period*5)
prob_params.dt = Constant(tidal_period/100)
# The initial condition consists of three components: u_x, u_y and eta
# Note that we do not set all components to zero, as some components of the
# Jacobian of the quadratic friction term is non-differentiable.
prob_params.initial_condition = Expression(("1e-7", "0", eta_channel), t=Constant(0),
              amp=tidal_amplitude, omega=2*pi/tidal_period, g=9.81, H=H)

# Here we only set the necessary options. A full option list with its current
# values can be viewed with:

print prob_params

# Once the parameter have been set, we create the shallow water problem:

problem = SWProblem(prob_params)

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form. Again, we first ask for the default
# parameters, adjust them to our needs and then create the solver object.

sol_params = CoupledSWSolver.default_parameters()
sol_params.dump_period = -1
solver = CoupledSWSolver(problem, sol_params)

# Now we are ready to solve the problem.

# Compute vorticity by L2 projection
class VorticitySolver(object):
    def __init__(self, V):
        self.u = Function(V)
        Q = V.extract_sub_space([0]).collapse()

        r = TrialFunction(Q)
        s = TestFunction(Q)
        a = r*s*dx
        self.L = (self.u[0].dx(1) - self.u[1].dx(0))*s*dx
        self.a_mat = assemble(a)

        self.vort = Function(Q)

    def solve(self, u):
        self.u.assign(u)
        L_mat = assemble(self.L)
        solve(self.a_mat, self.vort.vector(), L_mat, annotate=False)
        return self.vort

# Create output files
output_dir = "outputs"
u_xdmf = XDMFFile(mpi_comm_world(), "{}/u.xdmf".format(output_dir))
eta_xdmf = XDMFFile(mpi_comm_world(), "{}/eta.xdmf".format(output_dir))
vort_xdmf = XDMFFile(mpi_comm_world(), "{}/vorticity.xdmf".format(output_dir))

u_xdmf.parameters["rewrite_function_mesh"] = False
u_xdmf.parameters["flush_output"] = True
eta_xdmf.parameters["rewrite_function_mesh"] = False
eta_xdmf.parameters["flush_output"] = True
vort_xdmf.parameters["rewrite_function_mesh"] = False
vort_xdmf.parameters["flush_output"] = True

initialised = False
for s in solver.solve(annotate=False):
    print "Computed solution at time %f" % s["time"]
    if not initialised:
        initialised = True
        V = s["state"].function_space().extract_sub_space([0]).collapse()
        Q = s["state"].function_space().extract_sub_space([1]).collapse()
        vort_solver = VorticitySolver(V)

        u = Function(V)
        eta = Function(Q)

    u.assign(project(s["u"]), V)
    eta.assign(project(s["eta"]), Q)
    vort = vort_solver.solve(u)

    u_xdmf << u, float(s["time"])
    eta_xdmf << eta, float(s["time"])
    vort_xdmf << vort, float(s["time"])

# The inner part of the loop is executed for each timestep. The variable :attr:`s`
# is a dictionary and contains information like the current timelevel, the velocity and
# free-surface functions.

# The example code can be found in ``examples/channel-simulation/`` in the
# ``OpenTidalFarm`` source tree, and executed as follows:

# .. code-block:: bash

#   $ python channel-simulation.py
