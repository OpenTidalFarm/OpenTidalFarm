#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _channel_optimization:
#
# .. py:currentmodule:: opentidalfarm
#
# Farm layout optimization
# ========================
#
#
# Introduction
# ************
#

# Implementation
# **************
#

# The initial part of the program defines a steady state shallow water problem,
# and is nearly identical to the :ref:`channel_simulation` example:

from opentidalfarm import *

# Create a rectangular domain.
domain = FileDomain("mesh/mesh.xml")

# Specify boundary conditions.
bcs = BoundaryConditionSet()
bcs.add_bc("u", Constant((2, 0)), facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)
# The free-slip boundary conditions.
bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="weak_dirichlet")

# Set the shallow water parameters
prob_params = SteadySWProblem.default_parameters()
prob_params.domain = domain
prob_params.bcs = bcs
prob_params.viscosity = Constant(3)
prob_params.depth = Constant(50)
prob_params.friction = Constant(0.0025)

# The next step is to create the turbine farm. In this case, the
# farm consists of 32 turbines, initially deployed in a regular grid layout.
# This layout will be the starting guess for the optimization.

# Before adding turbines we must specify the type of turbines used in the array.
# Here we used the default BumpTurbine which defaults to being controlled by
# just position. The diameter and friction are set. The minimum distance between
# turbines if not specified is set to 1.5*diameter.
turbine = BumpTurbine(diameter=20.0, friction=12.0)

# A rectangular farm is defined using the domain and the site dimensions.
farm = RectangularFarm(domain, site_x_start=160, site_x_end=480,
                       site_y_start=80, site_y_end=240, turbine=turbine)

# Turbines are then added to the site in a regular grid layout.
farm.add_regular_turbine_layout(num_x=8, num_y=4)

prob_params.tidal_farm = farm

# Now we can create the shallow water problem

problem = SteadySWProblem(prob_params)

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form:

sol_params = CoupledSWSolver.default_parameters()
sol_params.dump_period = -1
solver = CoupledSWSolver(problem, sol_params)

# Next we create a reduced functional, that is the functional considered as a
# pure function of the control by implicitly solving the shallow water PDE.

functional = PowerFunctional
control = TurbineFarmControl(farm)
rf_params = ReducedFunctional.default_parameters()
rf_params.automatic_scaling = 5
rf = ReducedFunctional(functional, control, solver, rf_params)

# As always, we can print all options of the :class:`ReducedFunctional` with:

print rf_params

rf.derivative(...)
