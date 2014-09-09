..  #!/usr/bin/env python 
  # -*- coding: utf-8 -*-
  
.. _scenario1:

.. py:currentmodule:: opentidalfarm

Tidal simulation in the Orkney island
=====================================


Introduction
************

This example demonstrates how OpenTidalFarm can be used for simulating the
tides in a realistic domain.

Implementation
**************

We begin with importing the OpenTidalFarm module.

::

  from opentidalfarm import *
  
We also need the datetime module for the tidal forcing.

::

  import datetime
  
Next we define the UTM zone and band, as we we will need it multiple times
later on.

::

  utm_zone = 30
  utm_band = 'V'
  
We load the mesh and boundary ids from file   

::

  domain = FileDomain("../data/meshes/orkney/orkney_utm.xml")
  
We inspect the mesh and boundary ids with

::

  #plot(domain.facet_ids, interactive=True)
  
  
Next we specify boundary conditions. We apply tidal boundary forcing, by using
the :class:`TidalForcing` class.

::

  eta_expr = TidalForcing(grid_file_name='../data/netcdf/gridES2008.nc',
                          data_file_name='../data/netcdf/hf.ES2008.nc',
                          ranges=((-4.0,0.0), (58.0,61.0)),
                          utm_zone=utm_zone, 
                          utm_band=utm_band, 
                          initial_time=datetime.datetime(2001, 9, 18, 0),
                          constituents=['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2'])
  
  bcs = BoundaryConditionSet()
  bcs.add_bc("eta", eta_expr, facet_id=1)
  bcs.add_bc("eta", eta_expr, facet_id=2)
  
The free-slip boundary conditions are a special case. The boundary condition
type `weak_dirichlet` enforces the boundary value *only* in the
*normal* direction of the boundary. Hence, a zero weak Dirichlet
boundary condition gives us free-slip, while a zero `strong_dirichlet` boundary
condition would give us no-slip.

::

  bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="strong_dirichlet")
  
Next we load the bathymetry from the NetCDF file.

::

  bathy_expr = BathymetryDepthExpression('../data/netcdf/bathymetry.nc', utm_zone=utm_zone, 
                                    utm_band=utm_band)
  
The bathymetry can be visualised with

::

  #plot(bathy_expr, mesh=domain.mesh, title="Bathymetry", interactive=True)
  
Now we create shallow water problem and attach the domain and boundary
conditions

::

  prob_params = SWProblem.default_parameters()
  # Add domain and boundary conditions
  prob_params.domain = domain
  prob_params.bcs = bcs
  # Equation settings
  prob_params.viscosity = Constant(200)
  prob_params.depth = bathy_expr
  prob_params.friction = Constant(0.0025)
  # Temporal settings
  prob_params.start_time = Constant(0)
  prob_params.finish_time = Constant(12.5*60*60)
  prob_params.dt = Constant(60)
  # The initial condition consists of three components: u_x, u_y and eta
  # Note that we do not set all components to zero, as some components of the
  # Jacobian of the quadratic friction term is non-differentiable.
  prob_params.initial_condition_u = Constant((DOLFIN_EPS, 0)) 
  prob_params.initial_condition_eta = Constant(0) 
  # Create the shallow water problem
  problem = SWProblem(prob_params)
  
  # Next we create a shallow water solver. Here we choose to solve the shallow
  # water equations in its fully coupled form:
  sol_params = IPCSSWSolver.default_parameters()
  sol_params.dump_period = -1
  solver = IPCSSWSolver(problem, sol_params)
  
Now we are ready to solve

::

  for s in solver.solve():
      print "Computed solution at time %f" % s["time"]
      plot(s["eta"])
  
  # Finally we hold the plot unti the user presses q.
  interactive()
