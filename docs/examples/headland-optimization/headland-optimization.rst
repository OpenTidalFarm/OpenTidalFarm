..  #!/usr/bin/env python
  # -*- coding: utf-8 -*-
  
.. _headland_optimization:

.. py:currentmodule:: opentidalfarm

Continuous farm optimization
============================


Introduction
************

This example runs a tidal farm optimization based on a continuous turbine farm
representation. As outputs, one obtains an optimal turbine density function,
from which one can derive the optimal number of turbines to be deployed, and
their approximate layout within the farm.

It shows how to

- set up a time-dependent shallow water model with a continuous farm representation;
- define a callback function that is called after every optimisation iteration;
- run the optimization
- extract the optimal turbine density function and compute the optimal number of turbines.

The equations to be solved are the shallow water equations

.. math::
      \frac{\partial u}{\partial t} +  u \cdot \nabla  u - \nu \nabla^2 u  + g \nabla \eta + \frac{c_b}{H} \| u \|  u = 0, \\
      \frac{\partial \eta}{\partial t} + \nabla \cdot \left(H u \right) = 0, \\

where

- :math:`u` is the velocity,
- :math:`\eta` is the free-sufface displacement,
- :math:`H=\eta + h` is the total water depth where :math:`h` is the
  water depth at rest,
- :math:`c_b` is the (quadratic) natural bottom friction coefficient,
- :math:`\nu` is the viscosity coefficient,
- :math:`g` is the gravitational constant.


Implementation
**************


We begin with importing the OpenTidalFarm module.

::

  import argparse
  from opentidalfarm import *
  from model_turbine import ModelTurbine
  from vorticity_solver import VorticitySolver
  import time
  
  model_turbine = ModelTurbine()
  print model_turbine
  
  # Read the command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--turbines', required=True, type=int, help='number of turbines')
  parser.add_argument('--optimize', action='store_true', help='Optimise instead of just simulate')
  parser.add_argument('--withcuts', action='store_true', help='with cut in/out speeds')
  parser.add_argument('--cost', type=float, default=0., help='the cost coefficient')
  args = parser.parse_args()
  
Next we get the default parameters of a shallow water problem and configure it
to our needs.

::

  prob_params = SWProblem.default_parameters()
  
First we define the computational domain. We load a previously generated mesh
(using ```Gmsh``` and converted with ```dolfin-convert```):

::

  domain = FileDomain("mesh/headland.xml")
  
Once the domain is created we attach it to the problem parameters:

::

  prob_params.domain = domain
  
  turbine = SmearedTurbine()
  V = FunctionSpace(domain.mesh, "DG", 0)
  farm = Farm(domain, turbine, function_space=V)
  
  # Sub domain for inflow (right)
  class FarmDomain(SubDomain):
      def inside(self, x, on_boundary):
          return (9200 <= x[0] <= 10800 and
                  2500  <= x[1] <= 3500)
  
  farm_domain = FarmDomain()
  domains = MeshFunction("size_t", domain.mesh, domain.mesh.topology().dim())
  domains.set_all(0)
  farm_domain.mark(domains, 1)
  site_dx = Measure("dx")[domains]
  farm.site_dx = site_dx(1)
  #plot(domains, interactive=True)
  
  prob_params.tidal_farm = farm
  
Next we specify the boundary conditions.  The parameter `t` in the
:class:`dolfin.Expression` is special in OpenTidalFarm as it will be
automatically updated to the current timelevel during the solve.

::

  tidal_amplitude = 5.
  tidal_period = 12.42*60*60
  H = 40
  
  bcs = BoundaryConditionSet()
  eta_channel = "amp*sin(omega*t + omega/pow(g*H, 0.5)*x[0])"
  eta_expr = Expression(eta_channel, t=Constant(0), amp=tidal_amplitude,
                        omega=2*pi/tidal_period, g=9.81, H=H)
  bcs.add_bc("eta", eta_expr, facet_id=1, bctype="strong_dirichlet")
  bcs.add_bc("eta", eta_expr, facet_id=2, bctype="strong_dirichlet")
  
Free-slip boundary conditions need special attention. The boundary condition
type `weak_dirichlet` enforces the boundary value *only* in the
*normal* direction of the boundary. Hence, a zero weak Dirichlet
boundary condition gives us free-slip, while a zero `strong_dirichlet` boundary
condition would give us no-slip.

::

  bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="strong_dirichlet")
  
Again we attach boundary conditions to the problem parameters:

::

  prob_params.bcs = bcs
  
The other parameters are straight forward:

::

  # Equation settings
  nu = Constant(60)
  prob_params.viscosity = nu
  prob_params.depth = Constant(H)
  prob_params.friction = Constant(0.0025)
  # Temporal settings
  prob_params.theta = Constant(0.6)
  prob_params.start_time = Constant(0)
  prob_params.finish_time = Constant(2*tidal_period)
  prob_params.dt = Constant(tidal_period/100)
  prob_params.functional_final_time_only = False
  # The initial condition consists of three components: u_x, u_y and eta
  # Note that we do not set all components to zero, as some components of the
  # Jacobian of the quadratic friction term is non-differentiable.
  prob_params.initial_condition = Expression(("1e-7", "0", eta_channel), t=Constant(0),
                amp=tidal_amplitude, omega=2*pi/tidal_period, g=9.81, H=H)
  
Here we only set the necessary options. A full option list with its current
values can be viewed with:

::

  print prob_params
  
Once the parameter have been set, we create the shallow water problem:

::

  problem = SWProblem(prob_params)
  
Next we create a shallow water solver. Here we choose to solve the shallow
water equations in its fully coupled form. Again, we first ask for the default
parameters, adjust them to our needs and then create the solver object.

::

  sol_params = CoupledSWSolver.default_parameters()
  sol_params.dump_period = 1
  sol_params.output_dir = "output_{}_turbines_optimize_{}_cutinout_{}_cost_{}".format(args.turbines,
          args.optimize, args.withcuts, args.cost)
  sol_params.cache_forward_state = False
  solver = CoupledSWSolver(problem, sol_params)
  
  V = solver.function_space.extract_sub_space([0]).collapse()
  Q = solver.function_space.extract_sub_space([1]).collapse()
  
  base_u = Function(V, name="base_u")
  base_u_tmp = Function(V, name="base_u_tmp")
  
  # Define the functional
  if args.withcuts:
      power_functional = PowerFunctional(problem, cut_in_speed=1.0, cut_out_speed=3.)
  else:
      power_functional = PowerFunctional(problem)
  cost_functional = args.cost * CostFunctional(problem)
  functional = power_functional - cost_functional
  
  # Define the control
  control = TurbineFarmControl(farm)
  
  # Set up the reduced functional
  rf_params = ReducedFunctional.default_parameters()
  rf_params.automatic_scaling = None
  if args.optimize:
      rf_params.save_checkpoints = True
      rf_params.load_checkpoints = True
  
  rf = ReducedFunctional(functional, control, solver, rf_params)
  
As always, we can print all options of the :class:`ReducedFunctional` with:

::

  print rf_params
  
Now we can define the constraints for the controls and start the
optimisation.

::

  init_tf = model_turbine.maximum_smeared_friction/1000*args.turbines
  farm.friction_function.assign(Constant(init_tf))
  
  # Comment this for only forward modelling
  if args.optimize:
      maximize(rf, bounds=[0, model_turbine.maximum_smeared_friction],
              method="L-BFGS-B", options={'maxiter': 15})
  
  # Recompute the energy for the optimal farm array and store the results
  sol_h5 = HDF5File(mpi_comm_world(), "{}/solution.h5".format(sol_params.output_dir), "w")
  
  vort_solver = VorticitySolver(V)
  
  def callback(s):
      print "*** Storing timestep to solution.h5 ***"
      u = project(s["u"], V)
      eta = project(s["eta"], Q)
      vort = vort_solver.solve(u)
  
      sol_h5.write(u, "u_{}".format(float(s["time"])))
      sol_h5.write(eta, "eta_{}".format(float(s["time"])))
      sol_h5.write(vort, "vorticity_{}".format(float(s["time"])))
      sol_h5.write(farm.friction_function, "turbine_friction_{}".format(float(s["time"])))
      sol_h5.flush()
  
      total_friction = assemble(farm.friction_function*farm.site_dx(1))
      num_turbines = total_friction/model_turbine.friction
      print "Estimated number of turbines: ", float(num_turbines)
  
  # Recompute the cost, but this time with the power functional only
  j = rf(farm.control_array)
  
  # Power functional only
  solver.parameters.callback = callback
  rf_params.save_checkpoints = False
  rf_params.load_checkpoints = False
  rf = ReducedFunctional(power_functional, control, solver, rf_params)
  energy = rf(farm.control_array)
  
  # We are done with the solution.h5 file. Close it.
  sol_h5.close()
  
  # Save optimal friction as xdmf
  optimal_turbine_friction_file = XDMFFile(mpi_comm_world(),
       sol_params.output_dir+"/optimal_turbine_friction.xdmf")
  optimal_turbine_friction_file << farm.friction_function
  
  # Compute the total turbine friction
  total_friction = assemble(farm.friction_function*farm.site_dx(1))
  
  # Compute the total cost
  cost = float((prob_params.finish_time-prob_params.start_time) * args.cost * total_friction)
  
  # Compute the site area
  site_area = assemble(Constant(1)*farm.site_dx(1, domain=domain.mesh))
  
  avg_power = energy/1e6/float(prob_params.finish_time-prob_params.start_time)
  num_turbines = total_friction/model_turbine.friction
  
  print "="*40
  print "Site area (m^2): ", site_area
  print "Cost coefficient: {}".format(args.cost)
  print "Total energy (MWh): %e." % (energy/1e6/60/60)
  print "Average power (MW): %e." % avg_power
  print "Total cost: %e." % cost
  print "Maximum smeared turbine friction: %e." % model_turbine.maximum_smeared_friction
  print "Total turbine friction: %e." % total_friction
  print "Average smeared turbine friction: %e." % (total_friction / site_area)
  print "Average power / total friction: %e." % (avg_power / total_friction)
  print "Friction per discrete turbine: {}".format(model_turbine.friction)
  print "Estimated number of discrete turbines: {}".format(num_turbines)
