#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _channel_sensitivities:
#
# .. py:currentmodule:: opentidalfarm
#
# Farm Sensitivity Analysis
# =========================
#
#
# Introduction
# ************
#

# Gradient information may also be used to analyse the sensitivity of the chosen
# functional to various model parameters - for example the bottom friction.
# This enables the designer to identify which parameters may have a high impact
# upon the quality of the chosen design (as judged by the choice of functional).

# Implementation
# **************
#

# As with other examples, we begin by defining a steady state shallow water
# problem, once more this is nearly identical to the :ref:`channel_simulation`
# example except that we define steady flow:

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
prob_params.depth = interpolate(Constant(50), FunctionSpace(domain.mesh, "CG", 1))
prob_params.friction = Constant(0.0025)

# The next step is to specify the array design for which we wish to analyse
# the sensitivity. For simplicity we will use the starting guess from the
# :ref:`channel_optimization` example; 32 turbines in a regular grid layout.
# As before we'll use the default turbine type and define the diameter and
# friction. In practice, one is likely to want to analyse the sensitivity of
# the optimised array layout - so one would substitue this grid layout with
# the optimised one.

turbine = BumpTurbine(diameter=20.0, friction=12.0)
farm = RectangularFarm(domain, site_x_start=160, site_x_end=480,
                       site_y_start=80, site_y_end=240, turbine=turbine)
farm.add_regular_turbine_layout(num_x=8, num_y=4)
prob_params.tidal_farm = farm

# Now we can create the shallow water problem

problem = SteadySWProblem(prob_params)

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form:

sol_params = CoupledSWSolver.default_parameters()
sol_params.dump_period = -1
solver = CoupledSWSolver(problem, sol_params)

# We wish to study the effect that various model parameters have on the
# power. Thus, we select the :class:`PowerFunctional`


# Compupting derivatives with respect to farm controls
functional = PowerFunctional

# First let's print the sensitivity of the power with respect to the turbine
# positions. So we set the control to the turbine positions and intialise
# a reduced functional

control = TurbineFarmControl(farm)
rf_params = ReducedFunctional.default_parameters()
rf = ReducedFunctional(functional, control, solver, rf_params)
m0 = rf.solver.problem.parameters.tidal_farm.control_array
j = rf.evaluate(m0)
turbine_location_sensitivity = rf.derivative(m0)

print turbine_location_sensitivity

print "j for turbine positions: ", j

# Compute the sensitivity with respect to *other* controls
control = Control(prob_params.depth)
rf = FenicsReducedFunctional(functional, control, solver)

j = rf.evaluate()
dj = rf.derivative()

print "j with depth = 50 m: ", j
plot(dj, interactive=True)

# Update depth
prob_params.depth.assign(Constant(10))

j = rf.evaluate()
dj = rf.derivative()

print "j with depth = 10 m: ", j
plot(dj, interactive=True)
