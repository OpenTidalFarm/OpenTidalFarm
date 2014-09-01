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

# We begin with importing the OpenTidalFarm module.

from opentidalfarm import *

# Next we create a rectangular domain.

domain = FileDomain("mesh/mesh.xml")

# Next we specify boundary conditions. If a boundary expression contains a
# parameter named `t`, it will be automatically be updated to the current
# timelevel during the solve.

bcs = BoundaryConditionSet()
bcs.add_bc("u", Constant((2, 0)), facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)

# The free-slip boundary conditions are a special case. The boundary condition
# type `weak_dirichlet` enforces the boundary value *only* in the
# *normal* direction of the boundary. Hence, a zero weak Dirichlet
# boundary condition gives us free-slip, while a zero `strong_dirichlet` boundary
# condition would give us no-slip.

bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="weak_dirichlet")

# Now we create shallow water problem and attach the domain and boundary
# conditions

prob_params = SteadySWProblem.default_parameters()
# Add domain and boundary conditions
prob_params.domain = domain
prob_params.bcs = bcs
# Equation settings
prob_params.viscosity = Constant(3)
prob_params.depth = Constant(20)
prob_params.friction = Constant(0.0)
# Create the shallow water problem
problem = SteadySWProblem(prob_params)

# Here we set only the necessary options. However, there are many more,
# such as the `viscosity`, and the `bottom drag`. A full option list 
# with its current values can be viewed with:

print prob_params

# Next we create the turbine farm and deploy 32 turbines in a regular grid
# layout. This layout will be the starting guess for the optimization.

basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = DefaultConfiguration(domain)
config.set_site_dimensions(site_x_start, site_x_start+site_x, 
                           site_y_start, site_y_start+site_y)

deploy_turbines(config, nx=8, ny=4)


# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form:

sol_params = CoupledSWSolver.default_parameters()
sol_params.dump_period = -1
solver = CoupledSWSolver(problem, sol_params, config)

rf = ReducedFunctional(config, solver)

lb, ub = position_constraints(config) 
lb = [float(l) for l in lb]
ub = [float(u) for u in ub]
ineq = get_minimum_distance_constraint_func(config)
f_opt = maximize(rf, bounds=[lb, ub], method="SLSQP", options={'maxiter':3})

plot(f_opt, interactive=True)
