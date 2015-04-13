#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _farm-performance:
#
# .. py:currentmodule:: opentidalfarm
#
# Farm power production
# =====================
#
# Introduction
# ************
#
# This demo estimates the power extraction of a tidal farm with 32 turbines.
# It shows how to
#
# - set up a rectangular domain
# - define a tidal farm and deploy turbines
# - solve the steady-state shallow water solver
# - compute the power extraction of the farm

# Even though the domain in this demo is quite simple, the concept applies to
# more complex, realistic time-dependent scenarios.
#
# The turbines are deployed in staggered layout:
#
# .. image:: 32turbines_staggered.png
#     :scale: 30
#
# The computed flow speed with streamlines is:
#
# .. image:: 32turbines_staggered_vel.png
#     :scale: 30
#
# The power extraction by the farm (without taking losses due to wake effects
# and electrical losses into account) is 64 MW or 2.0 MW per turbine.
#
# Note: In the next example we will see how OpenTidalFarm can optimies the
# position of the turbines to improve the performance of the farm.

# Implementation
# **************
#

# The first part of the program sets up a steady state shallow water problem,
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
prob_params.viscosity = Constant(2)
prob_params.depth = Constant(50)
prob_params.friction = Constant(0.0025)

# The next step is to create the turbine farm. In this case, the
# farm consists of 32 turbines deployed in a staggered grid layout.

# We start by specifying the turbine type used in the array.
# Here we use BumpTurbine, which represents turbines as
# increased bottom friction in the shallow water equations in the shape of a
# bump function:

# .. image:: turbine_bump.jpg

# We need to specify the diameter and the peak friction of the turbine
# representation:

turbine = BumpTurbine(diameter=20.0, friction=12.0)

# Next we create a farm in which we can deploy the turbines. A rectangular farm
# is defined using the domain and the site dimensions.

farm = RectangularFarm(domain, site_x_start=160, site_x_end=480,
                       site_y_start=80, site_y_end=240, turbine=turbine)

# We could also add turbines manually via the
# :class:`RectangularFarm.add_turbine`.
# For simplicity, we use a helper functions to deploy 32 turbines in a staggered grid layout.

farm.add_staggered_turbine_layout(num_x=8, num_y=4)
prob_params.tidal_farm = farm

# We can plot the friction function produced by this farm with

plot(farm.friction_function, title="Farm friction")

# .. image:: farm_friction.png
#     :scale: 30

# Now we can create the shallow water problem

problem = SteadySWProblem(prob_params)

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form. We also set the dump period to 1 in
# order to save the results of each optimisation iteration to disk.

sol_params = CoupledSWSolver.default_parameters()
solver = CoupledSWSolver(problem, sol_params)

# Next we need to define the objective functional, i.e. the value that we want
# to compute. In this case we are interested in the power extracted from the
# farm, hence we use the :class:`PowerFunctional` functional.

functional = PowerFunctional(problem)

# The next few lines are mostly relevant for optimisation, but we need them here
# anyway.

control = TurbineFarmControl(farm)
rf_params = ReducedFunctional.default_parameters()
rf_params.automatic_scaling = None
rf = ReducedFunctional(functional, control, solver, rf_params)

# Now we can evaluate the power production of the farm.

power = rf(farm.control_array)
print "Extracted power by farm is {} MW.".format(power/1e6)

# How to run the example
# **********************

# The example code can be found in ``examples/farm-performance/`` in the
# ``OpenTidalFarm`` source tree, and executed as follows:

# .. code-block:: bash

#   $ python farm-performance.py
