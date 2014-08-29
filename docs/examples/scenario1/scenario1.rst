..  #!/usr/bin/env python 
  # -*- coding: utf-8 -*-
  
.. _scenario1:

.. py:currentmodule:: opentidalfarm

Turbine farm layout optimisation in a channel
=============================================


Background
**********


::

  from opentidalfarm import *
  set_log_level(INFO)
  
  # Some domain information extracted from the geo file
  inflow_direction = [1, 0]
  basin_x = 640.
  basin_y = 320.
  site_x = 320.
  site_y = 160.
  site_x_start = (basin_x - site_x)/2
  site_y_start = (basin_y - site_y)/2 
  config = SteadyConfiguration("mesh_coarse.xml",
          inflow_direction=inflow_direction)
  config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start,
                             site_y_start + site_y)
  
  # Place some turbines 
  deploy_turbines(config, nx=8, ny=4)
  
  config.info()
  
  parameters = SteadySWProblem.default_parameters()
  
  bc = DirichletBCSet(config)
  bc.add_constant_flow(1, 2.0 + 1e-10, direction=inflow_direction)
  bc.add_analytic_eta(2, 0.0)
  parameters["strong_bc"] = bc
  
  problem = SteadySWProblem(parameters)
  
  parameters = SWSolver.default_parameters()
  solver = SWSolver(problem, parameters, config)
  
  rf = ReducedFunctional(config, solver)
  
  lb, ub = position_constraints(config) 
  ineq = get_minimum_distance_constraint_func(config)
  maximize(rf, bounds=[lb, ub], constraints=ineq, method="SLSQP",
           options={"maxiter": 3})
