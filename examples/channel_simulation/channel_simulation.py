#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# .. _scenario1:
#
# .. py:currentmodule:: opentidalfarm
#
# A simulation of a sinusoidal wave with the shallow water equations
# ==================================================================
#
#
# Background
# **********
#

from opentidalfarm import *

# Read in mesh
domain = RectangularDomain(x0=0, y0=0, x1=100, y1=100, nx=25, ny=10)

# You can plot and inspect the boundary ids with
#plot(domain.facet_ids, interactive=True)

# Specify boundary conditions
bcs = BoundaryConditionSet()
u_expr = Expression(("sin(t)", "0"), t=Constant(0))
bcs.add_bc("u", u_expr, facet_id=1, bctype="weak_dirichlet")
bcs.add_bc("eta", Constant(0), facet_id=2)
# Free-slip on the sides
bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="weak_dirichlet")

# Create shallow water problem
params = ShallowWaterProblem.default_parameters()
params.domain = domain
params.bcs = bcs
problem = ShallowWaterProblem(params)

# You can view the current settings with
print params

# Create shallow water solver
params = ShallowWaterSolver.default_parameters()
params.dump_period = -1
solver = ShallowWaterSolver(problem, params)

for s in solver.solve():
    print "Computed solution at time %f" % s["time"]
    plot(s["state"])
