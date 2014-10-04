#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _channel_optimization:
#
# .. py:currentmodule:: opentidalfarm
#
# Layout optimization of 32 tidal turbines in a channel
# =====================================================
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

# A rectangular farm is defined using the domain and the site dimensions.
farm = RectangularFarm(domain,
                       site_x_start=160, site_x_end=480,
                       site_y_start=80, site_y_end=240)

# Before adding turbines we must specify the type of turbines used in the array,
# their controls and their parameterisation.
controls = Controls(position=True)
parameterisation = TurbineParameterisation()
farm.turbine_prototype = Turbine(diameter=20.0, minimum_distance=40.0,
                                 maximum_friction=21.0, controls=controls,
                                 parameterisation=parameterisation)

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
rf_params = ReducedFunctional.default_parameters()
rf_params.automatic_scaling = 5
rf = ReducedFunctional(functional, solver, rf_params)

# As always, we can print all options of the :class:`ReducedFunctional` with:

print rf_params

# Finally, we can define the constraints for the controls and start the
# optimisation.

lb, ub = farm.site_boundary_constraints()
ineq = farm.minimum_distance_constraints()

f_opt = maximize(rf, bounds=[lb, ub], method="L-BFGS-B", options={'maxiter':10})

print "Optimised turbine positions: ", f_opt

# FIXME: This should be accessible more easily
plot(farm.turbine_cache.cache["turbine_field"], interactive=True)
