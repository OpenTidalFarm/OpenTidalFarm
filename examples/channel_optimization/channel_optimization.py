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

# Create shallow water problem 
prob_params = SteadySWProblem.default_parameters()
prob_params.domain = domain
prob_params.bcs = bcs
prob_params.viscosity = Constant(3)
prob_params.depth = Constant(50)
prob_params.friction = Constant(0.0025)
problem = SteadySWProblem(prob_params)

# After that, the next step is to create the turbine farm. In this case, the
# farm consists of 32 turbines, initialloy deployed in a regular grid layout.
# This layout will be the starting guess for the optimization.

farm = DefaultConfiguration(domain)
farm.set_site_dimensions(x0=160, x1=480, 
                         y0=80, y1=240)
farm.params['controls'] = ['turbine_pos']
farm.params['turbine_x'] = 20
farm.params['turbine_y'] = 20

deploy_turbines(farm, nx=8, ny=4)

farm.info()

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form:

sol_params = CoupledSWSolver.default_parameters()
sol_params.dump_period = -1
solver = CoupledSWSolver(problem, sol_params, farm)

functional = PowerFunctional
rf = ReducedFunctional(farm, functional, solver, automatic_scaling=5)

lb, ub = position_constraints(farm) 
ineq = get_minimum_distance_constraint_func(farm)
f_opt = maximize(rf, bounds=[lb, ub], method="L-BFGS-B", options={'maxiter':10})

print "Optimised turbine positions: ", f_opt

# FIXME: This should be accessible more easily
plot(farm.turbine_cache.cache["turbine_field"], interactive=True)
