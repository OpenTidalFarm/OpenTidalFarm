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

# The first part of the program is nearly identical to channel simulation
# example. FIXME: Add link.

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

# Now, the first step is to create the turbine farm and deploy 32 turbines in a
# regular grid layout. This layout will be the starting guess for the
# optimization.

basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
farm = DefaultConfiguration(domain)
farm.set_site_dimensions(site_x_start, site_x_start+site_x, 
                           site_y_start, site_y_start+site_y)
farm.params['controls'] = ['turbine_pos']
farm.params['turbine_x'] = 20
farm.params['turbine_y'] = 20
farm.params['automatic_scaling'] = True

deploy_turbines(farm, nx=8, ny=4)

farm.info()

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form:

sol_params = CoupledSWSolver.default_parameters()
sol_params.dump_period = -1
solver = CoupledSWSolver(problem, sol_params, farm)

rf = ReducedFunctional(farm, solver)

lb, ub = position_constraints(farm) 
ineq = get_minimum_distance_constraint_func(farm)
f_opt = maximize(rf, bounds=[lb, ub], method="L-BFGS-B", options={'maxiter':10})

print "Optimised turbine positions: ", f_opt

# FIXME: This should be accessible more easily
plot(farm.turbine_cache.cache["turbine_field"], interactive=True)
