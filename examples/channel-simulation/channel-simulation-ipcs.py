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
# It shows how to:
#   - create a rectangular domain;
#   - specify velocity and surface elevation boundary conditions;
#   - set up a and solve the shallow water solver;
#   - plot the results.
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

domain = RectangularDomain(x0=0, y0=0, x1=100, y1=50, nx=20, ny=10)

# The boundary of the domain is marked with integers in order to specify
# different boundary conditions on different parts of the domain. You can plot
# and inspect the boundary ids with:

plot(domain.facet_ids)

# Once the domain is created we attach it to the problem parameters:

prob_params.domain = domain

# Next we specify boundary conditions. For time-dependent boundary condition use
# a parameter named `t` in the :class:`dolfin.Expression` and it will be automatically be
# updated to the current timelevel during the solve.

bcs = BoundaryConditionSet()
u_expr = Expression(("sin(pi*t/5)*x[1]*(50-x[1])/625", "0"), t=Constant(0))
bcs.add_bc("u", u_expr, facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)

# Free-slip boundary conditions need special attention. The boundary condition
# type `weak_dirichlet` enforces the boundary value *only* in the
# *normal* direction of the boundary. Hence, a zero weak Dirichlet
# boundary condition gives us free-slip, while a zero `strong_dirichlet` boundary
# condition would give us no-slip.

bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="strong_dirichlet")

# Again we attach boundary conditions to the problem parameters:

prob_params.bcs = bcs

# The other parameters are straight forward:

# Equation settings
prob_params.viscosity = Constant(30)
prob_params.depth = Constant(20)
prob_params.friction = Constant(0.0)
# Temporal settings
prob_params.theta = Constant(0.5)
prob_params.start_time = Constant(0)
prob_params.finish_time = Constant(500)
prob_params.dt = Constant(0.5)
# The initial condition consists of three components: u_x, u_y and eta
# Note that we do not set all components to zero, as some components of the
# Jacobian of the quadratic friction term is non-differentiable.
prob_params.initial_condition_u = Constant((DOLFIN_EPS, 0))
prob_params.initial_condition_eta = Constant(0)
prob_params.linear_divergence = True

# Here we only set the necessary options. A full option list with its current
# values can be viewed with:

print prob_params

# Once the parameter have been set, we create the shallow water problem:

problem = SWProblem(prob_params)

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form. Again, we first ask for the default
# parameters, adjust them to our needs and then create the solver object.

sol_params = IPCSSWSolver.default_parameters()
solver = IPCSSWSolver(problem, sol_params)

# Now we are ready to solve the problem.

for s in solver.solve():
    print "Computed solution at time %f" % s["time"]
    plot(s["u"], title="u")
    plot(s["eta"], title="eta")
    plot(s["eddy_viscosity"], title="eddy viscosity")
interactive()  # Hold the plot until the user presses q.

# The inner part of the loop is executed for each timestep. The variable :attr:`s`
# is a dictionary and contains information like the current timelevel, the velocity and
# free-surface functions.

# How to run the example
# **********************

# The example code can be found in ``examples/channel-simulation/`` in the
# ``OpenTidalFarm`` source tree, and executed as follows:

# .. code-block:: bash

#   $ python channel-simulation.py
