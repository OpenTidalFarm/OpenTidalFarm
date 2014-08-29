#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# .. _scenario1:
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
# in-/outflow on the west side, fixed surface height on the east side, and 
# free-slip flow on the north and south sides.
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
# - :math:`H=\eta + depth` is the total water depth where :math:`depth` is the
#   water depth at rest,
# - :math:`c_b` is the (quadratic) natural bottom friction coefficient,
# - :math:`\nu` is the viscosity coefficient,
# - :math:`g` is the gravitational constant.
#
# The boundary conditions are:
#
# .. math::
#       u = \sin(2 \pi t/60) & \quad \textrm{on } \Gamma_1, \\
#       \eta = 0 & \quad \textrm{on } \Gamma_2, \\
#       u \cdot n = 0 & \quad \textrm{on } \Gamma_3, \\
#
# where :math:`n` is the normal vector pointing outside the domain,
# :math:`\Gamma_1` is the west boundary of the channel, :math:`\Gamma_2` is the
# east boundary of the channel, and :math:`\Gamma_3` is the north and south
# boundaries of the channel.
#
#

# Implementation
# **************
#

# We begin with importing the OpenTidalFarm module.

from opentidalfarm import *

# Next we create a rectangular domain.

domain = RectangularDomain(x0=0, y0=0, x1=100, y1=50, nx=10, ny=5)

# You can plot and inspect the boundary ids with

plot(domain.facet_ids, interactive=True)

# Next we specify boundary conditions. If a boundary expression contains a
# parameter named `t`, it will be automatically be updated to the current
# timelevel during the solve.

bcs = BoundaryConditionSet()
u_expr = Expression(("sin(2*pi*t/60)", "0"), t=Constant(0))
bcs.add_bc("u", u_expr, facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)

# The free-slip boundary conditions are a special case. The boundary condition
# type `weak_dirichlet` enforces the boundary value *only* in the
# *normal* direction of the boundary. Hence, a zero weak Dirichlet
# boundary condition gives us free-slip, while a zero `strong_dirichlet` boundary
# condition would give us no-slip.

bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="weak_dirichlet")

# Now we create shallow water problem and attach the domain and boundary
# conditions

prob_params = SWProblem.default_parameters()
# Add domain and boundary conditions
prob_params.domain = domain
prob_params.bcs = bcs
# Equation settings
prob_params.viscosity = Constant(3)
prob_params.depth = Constant(20)
prob_params.friction = Constant(0.0)
# Temporal settings
prob_params.start_time = Constant(0)
prob_params.finish_time = Constant(60)
prob_params.dt = Constant(6)
# The initial condition consists of three components: u_x, u_y and eta
# Note that we do not set all components to zero, as the 
prob_params.initial_condition = Constant((0, 0, 0)) 
# Create the shallow water problem
problem = SWProblem(prob_params)

# Here we set only the necessary options. However, there are many more,
# such as the `viscosity`, and the `bottom drag`. A full option list 
# with its current values can be viewed with:

print prob_params

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form:

sol_params = SWSolver.default_parameters()
sol_params.dump_period = -1
solver = SWSolver(problem, sol_params)

# Now we are ready to solve

for s in solver.solve():
    print "Computed solution at time %f" % s["time"]
    plot(s["state"])

# Finally we hold the plot unti the user presses q.

interactive()
