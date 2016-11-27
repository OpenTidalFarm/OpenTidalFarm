#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _resource_assessment:
#
# .. py:currentmodule:: opentidalfarm
#
# Resource assessment in the Orkney island
# ========================================
#
# Note: This example is unfinished
#
# Introduction
# ************
#
# This example demonstrates how OpenTidalFarm can be used for assessing the
# potential of a site in a realistic domain.
#
# We will be simulating the tides in the Pentland Firth, Scotland for 12.5
# hours, starting at 14:40 am on the 18.9.2001.
#
# This example uses a "continuous turbine approach", as described in
#    **Funke SW, Kramer SC, Piggott MD**, *Design optimisation and resource assessment
#    for tidal-stream renewable energy farms using a new continuous turbine
#    approach*

# To run this example, some data files must be downloaded
# separately by calling in the source code directory:
#
#
# .. code-block:: bash
#
#    git submodule init
#    git submodule update


# Implementation
# **************

# We begin with importing the OpenTidalFarm module.

from opentidalfarm import *

# We also need the datetime module for the tidal forcing.

import datetime

# Next we define the UTM zone and band, as we we will need it multiple times
# later on.

utm_zone = 30
utm_band = 'V'

# Next we create shallow water problem and attach the domain and boundary
# conditions

prob_params = SWProblem.default_parameters()

# We load the mesh in UTM coordinates, and boundary ids from file

domain = FileDomain("../data/meshes/orkney/orkney_utm.xml")
prob_params.domain = domain

# The mesh and boundary ids can be visualised with

#plot(domain.facet_ids, interactive=True)

# Next we specify boundary conditions. We apply tidal boundary forcing, by using
# the :class:`TidalForcing` class.

eta_expr = TidalForcing(grid_file_name='../data/netcdf/gridES2008.nc',
                        data_file_name='../data/netcdf/hf.ES2008.nc',
                        ranges=((-4.0,0.0), (58.0,61.0)),
                        utm_zone=utm_zone,
                        utm_band=utm_band,
                        initial_time=datetime.datetime(2001, 9, 18, 10, 40),
                        constituents=['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2'], degree=3)

bcs = BoundaryConditionSet()
bcs.add_bc("eta", eta_expr, facet_id=1)
bcs.add_bc("eta", eta_expr, facet_id=2)

# The free-slip boundary conditions are a special case. The boundary condition
# type `weak_dirichlet` enforces the boundary value *only* in the *normal*
# direction of the boundary. Hence, a zero weak Dirichlet boundary condition
# gives us free-slip, while a zero `strong_dirichlet` boundary condition would
# give us no-slip.

bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="strong_dirichlet")
prob_params.bcs = bcs

# Next we load the bathymetry from the NetCDF file.

bathy_expr = BathymetryDepthExpression(filename='../data/netcdf/bathymetry.nc',
        utm_zone=utm_zone, utm_band=utm_band, domain=domain.mesh, degree=3)
prob_params.depth = bathy_expr

# The bathymetry can be visualised with

#plot(bathy_expr, mesh=domain.mesh, title="Bathymetry", interactive=True)

# Equation settings

# For stability reasons, we want to increase the viscosity at the inflow and
# outflow boundary conditions. For that, we read in a precomputed function
# (generated by ```compute_distance```) and

V = FunctionSpace(domain.mesh, "CG", 1)
dist = Function(V)
File("dist.xml") >> dist

# With that we can define an expression that evaluates to a nu_inside value
# inside the domain and a nu_outside value near the in/outflow boundary.

class ViscosityExpression(Expression):
    def __init__(self, **kwargs):
        self.dist_function = kwargs["dist_function"]
        self.nu_inside = kwargs["nu_inside"]
        self.nu_boundary = kwargs["nu_boundary"]
        self.dist_threshold = kwargs["dist_threshold"]

    def eval(self, value, x):
        if self.dist_function(x) > self.dist_threshold:
            value[0] = self.nu_inside
        else:
            value[0] = self.nu_boundary

# Finally, we interpolate this expression to a piecewise discontinuous, constant
# function and attach it as the viscosity value to the shallow water problem.

W = FunctionSpace(domain.mesh, "DG", 0)
nu = ViscosityExpression(dist_function=dist, dist_threshold=10000., nu_inside=5000.,
        nu_boundary=1e4, degree=3)
nu_func = interpolate(nu, W)
prob_params.viscosity = nu_func

# The other parameters are set as usual.

prob_params.friction = Constant(0.0025)
# Temporal settings
prob_params.start_time = Constant(0)
prob_params.finish_time = Constant(20*60) # 12.5*60*60)
prob_params.dt = Constant(5*60)
# The initial condition consists of three components: u_x, u_y and eta.
# Note that we set the velocity components to a small positive number, as some
# components of the Jacobian of the quadratic friction term is
# non-differentiable.
prob_params.initial_condition = Constant((DOLFIN_EPS, DOLFIN_EPS, 1))

# We use the continuum turbine parametrisation
turbine = SmearedTurbine()
farm = Farm(domain, turbine)
prob_params.tidal_farm = farm

# Now we can create the shallow water problem
problem = SWProblem(prob_params)

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form:

sol_params = CoupledSWSolver.default_parameters()
solver = CoupledSWSolver(problem, sol_params)

#plot(farm.domain.cell_ids)


# Now we can define the functional and control values:

functional = PowerFunctional(problem)
control = TurbineFarmControl(farm)

# Finally, we create the reduced functional and start the optimisation.

rf_params = ReducedFunctionalParameters()
rf_params.automatic_scaling = False
rf = ReducedFunctional(functional, control, solver, rf_params)
f_opt = maximize(rf, bounds=[0, 0.05890486225480861],
                 method="L-BFGS-B", options={'maxiter': 30})

# The code for this example can be found in ``examples/resource-assessment/`` in the
# ``OpenTidalFarm`` source tree, and executed as follows:

# .. code-block:: bash

#   $ python resource-assessment.py
