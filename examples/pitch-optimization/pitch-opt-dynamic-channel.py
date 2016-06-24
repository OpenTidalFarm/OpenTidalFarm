#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Farm pitch optimization
# ========================
#
# This demo optimizes the friction coefficient of 1 turbine dynamicelly in a
# channel, which should be equivalent to changing the pitch of the turbines blades.
# The goal of the optimization is to maximise the farm's energy extraction. The
# rectangular channel is 640 m x 320 m large. The farm area is in the channel
# center and 320 m x 160 m large.

# It shows how to:
#   - set up a shallow water solver
#   - define an optimisation objective, here the farm's power production;
#   - define an optimisation control, here the turbine friction coefficients;
#   - define constraints to the controls, here bounds to restrict the positions
#     of the turbines to the farm bounds;
#   - run the optimisation and retrieve the optimal turbine friction
#   coefficents for each timestep for each turbine to produce the most power.

# Even though the domain in this demo is quite simple, the concept applies to
# more complex, realistic scenarios.
#
# The boundary conditions are free-slip conditions on the sides and constant 2
# ms^-1 inflow and free-surface displacement equal zero on the outflow.


from opentidalfarm import *
import os

# Create a rectangular domain.
domain = FileDomain("mesh/mesh.xml")

# Boundary conditions
bcs = BoundaryConditionSet()
bcs.add_bc("u", Constant((2, 0)), facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)
bcs.add_bc("u", facet_id=3, bctype="free_slip")
bcs.add_bc("u", facet_id=4, bctype="free_slip")

# Set the problem_params for the Shallow water problem
problem_params = SWProblem.default_parameters()
problem_params.bcs = bcs
problem_params.domain = domain

# Activate the relevant terms
problem_params.include_advection = True
problem_params.include_viscosity = True
problem_params.linear_divergence = False

# Physical settings
problem_params.friction = Constant(0.0025)
problem_params.viscosity = Constant(3.0)
problem_params.depth = Constant(50)
problem_params.g = Constant(9.81)

# Set time parameters
period = 12. * 60 * 60
problem_params.start_time = Constant(1. / 4 * period)
problem_params.dt = Constant(period / 500)
n_time_steps = 10
# The 10000*DOLFIN_EPS is to avoid machine precision errors 
problem_params.finish_time = problem_params.start_time + \
                             (n_time_steps-1) * problem_params.dt - 10000*DOLFIN_EPS

problem_params.functional_final_time_only = False

# Use Crank-Nicolson to get a second-order time-scheme
problem_params.theta = 1

# Create a turbine
turbine = BumpTurbine(diameter=20., friction=13,
					  controls=Controls(dynamic_friction=True))

# Adjust some global options
options["output_turbine_power"] = True

# Create Tidalfarm
n_turbines_x = 1
n_turbines_y = 1
basin_x = 640
basin_y = 320
site_x = 320
site_y = 160
if (n_turbines_x == 1):
    site_x = 40
if (n_turbines_y == 1):
    site_y = 40
turbine_pos = []
farm = RectangularFarm(domain,
					   site_x_start = basin_x/2 - site_x/2,
					   site_x_end = basin_x/2 + site_x/2,
					   site_y_start = basin_y/2 - site_y/2,
					   site_y_end = basin_y/2 + site_y/2,
					   turbine = turbine)

farm.add_regular_turbine_layout(num_x=n_turbines_x, num_y=n_turbines_y)

# Extend the friction array to hold all the time steps
friction = farm._parameters["friction"]
farm._parameters["friction"] = [friction]*(n_time_steps)

problem_params.tidal_farm = farm


# Create problem
problem = SWProblem(problem_params)
solver_params = CoupledSWSolver.default_parameters()
solver_params.dump_period = 1
solver_params.cache_forward_state = False
solver = CoupledSWSolver(problem, solver_params)

functional = PowerFunctional(problem)
control = TurbineFarmControl(farm)
rf_params = ReducedFunctionalParameters()
rf_params.scale = 10**-6
rf_params.automatic_scaling = False
rf = ReducedFunctional(functional, control, solver, rf_params)

# Now we can define the constraints for the controls and start the
# optimization.
lb, ub = farm.friction_constraints(n_time_steps=n_time_steps,
                                   lower_bounds=0., upper_bounds=500.)
f_opt = maximize(rf, bounds=[lb,ub],  method="L-BFGS-B", 
        options={'maxiter': 5,'ftol': 1.0e-04})
f_opt = f_opt.reshape((n_time_steps, n_turbines_x, n_turbines_y))

# Print the optimized friction coefficients for each timestep and each
# position. 
print f_opt

# How to run the example
# **********************

# The example code can be found in ``examples/pitch-optimization/`` in the
# ``OpenTidalFarm`` source tree, and executed as follows:

# .. code-block:: bash

#   $ python pitch-opt-dynamic-channel.py

# You can speed up the calculation by using multiple cores (in this case 4)
# with:

# .. code-block:: bash

#   $ mpirun -n 4 python pitch-opt-dynamic-channel.py

# During the optimization run, OpenTidalFarm creates multiple files for
# inspection:
#
# *  turbines.pvd: Stores the position and friction values of the turbines at
#    each optimisation iteration.
# *  iter_*: For each optimisation iteration X, the associated
#    velocity and pressure solutions are stored in a directory named iter_X.
# *  iterate.dat: A testfile that dumps the optimisation progress, e.g. number
# of
#    iterations, function value, gradient norm, etc
#
# The pvd files can be opened with the open-source software
# `Paraview <http://www.paraview.org>`_.
