#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# .. _scenario1:
#
# .. py:currentmodule:: opentidalfarm
#
# Tidal simulation in the Orkney island
# =====================================
#
#
# Introduction
# ************
#
# This example demonstrates how OpenTidalFarm can be used for simulating the
# tides in a realsitic domain.

# We begin with importing the OpenTidalFarm module.

from opentidalfarm import *
import datetime
from math import pi


# We load the mesh and boundary ids from file   

domain = FileDomain("../data/meshes/orkney/orkney_utm.xml")

# We inspect the mesh and boundary ids with

#plot(domain.facet_ids, interactive=True)


import sys; sys.exit()

# Next we specify boundary conditions. We want to use real tidal boundary 
# forcing, hence we use the :class:`TidalForcing` class.

bc = DirichletBCSet(config)
eta_expr = TidalForcing(grid_file_name='netcdf/gridES2008.nc',
                        data_file_name='netcdf/hf.ES2008.nc',
                        ranges=((-4.0,0.0), (58.0,61.0)),
                        utm_zone=utm_zone, 
                        utm_band=utm_band, 
                        initial_time=datetime.datetime(2001, 9, 18, 0),
                        constituents=['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2'])

bc.add_analytic_eta(1, eta_expr)
bc.add_analytic_eta(2, eta_expr)
# comment out if you want free slip:
#bc.add_noslip_u(3)
config.params['bctype'] = 'strong_dirichlet'
config.params['strong_bc'] = bc
bcs = BoundaryConditionSet()
u_expr = Expression(("sin(pi*t/60)", "0"), t=Constant(0))
bcs.add_bc("u", u_expr, facet_id=1)

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
prob_params.finish_time = Constant(120)
prob_params.dt = Constant(6)
# The initial condition consists of three components: u_x, u_y and eta
# Note that we do not set all components to zero, as some components of the
# Jacobian of the quadratic friction term is non-differentiable.
prob_params.initial_condition = Constant((DOLFIN_EPS, 0, 0)) 
# Create the shallow water problem
problem = SWProblem(prob_params)

# Here we set only the necessary options. However, there are many more,
# such as the `viscosity`, and the `bottom drag`. A full option list 
# with its current values can be viewed with:

print prob_params

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form:

sol_params = CoupledSWSolver.default_parameters()
sol_params.dump_period = -1
solver = CoupledSWSolver(problem, sol_params)

# Now we are ready to solve

for s in solver.solve():
    print "Computed solution at time %f" % s["time"]
    plot(s["state"])

# Finally we hold the plot unti the user presses q.
interactive()

utm_zone = 30
utm_band = 'V'
config.params['initial_condition'] = ConstantFlowInitialCondition(config) 
config.params['viscosity'] = 180.0
config.params["controls"] = ["turbine_friction"]
config.params["turbine_parametrisation"] = "smeared"
config.params["automatic_scaling"] = False 
config.params['friction'] = Constant(0.0025)

config.params['start_time'] = 0 
config.params['dt'] = 600 
config.params['finish_time'] = 12.5 * 60 * 60 
config.params['theta'] = 1.0 


V_cg1 = FunctionSpace(config.domain.mesh, "CG", 1)
V_dg0 = FunctionSpace(config.domain.mesh, 'DG', 0)

# Bathymetry
bexpr = BathymetryDepthExpression('netcdf/bathymetry.nc', utm_zone=utm_zone, utm_band=utm_band)
depth = interpolate(bexpr, V_cg1) 
depth_pvd = File(os.path.join(config.params["base_path"], "bathymetry.pvd"))
depth_pvd << depth

config.params['depth'] = depth
config.turbine_function_space = V_dg0 

domains = MeshFunction("size_t", config.domain.mesh, mesh_basefile + "_physical_region.xml")
if farm_selector is not None:
  domains_ids = MeshFunction("size_t", config.domain.mesh, mesh_basefile + "_physical_region.xml")
  domains.set_all(0)
  domains.array()[domains_ids.array() == farm_selector] = 1
#plot(domains, interactive=True)
config.site_dx = Measure("dx")[domains]
f = File(os.path.join(config.params["base_path"], "turbine_farms.pvd"))
f << domains

config.info()

rf = ReducedFunctional(config, scale=-1e-6)

print "Running forward model"
m0 = rf.initial_control()
rf.j(m0, annotate=False)
print "Finished"
