#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _channel_simulation:
#
# .. py:currentmodule:: opentidalfarm
#
# Dynamic optimization of friction coefficient 
# ============================
#
#
# Introduction
# ************
#
# This example optimizes the friction coefficient of 32 turbines in a channel
# with sinusoidal inflow velocity the west boundary, fixed surface height on
# the east boundary, and no-slip flow on the north and south boundaries.
#
# It shows how to:
#   - specify velocity and surface elevation boundary conditions;
#   - set up the shallow water solver;
#   - set up the redused functional for the energy output;
#   - optimize the friction coefficient dynamically for the energy output;
#
# The shallow water equations to be solved are
#
# .. math::
#       \frac{\partial u}{\partial t} +  u \cdot \nabla  u - \nu \nabla^2 u  + g \nabla \eta + \frac{c_b}{H} \| u \|  u = 0, \\
#       \frac{\partial \eta}{\partial t} + \nabla \cdot \left(H u \right) = 0, \\
#
# where
#
# - :math:`u` is the velocity,
# - :math:`\eta` is the free-surface displacement,
# - :math:`H=\eta + h` is the total water depth where :math:`h` is the
#   water depth at rest,
# - :math:`c_b` is the (quadratic) natural bottom friction coefficient,
# - :math:`\nu` is the viscosity coefficient,
# - :math:`g` is the gravitational constant.
#
# The boundary conditions are:
#
# .. math::
#       u = \begin{pmatrix}3*\sin(\frac{\pi t}{600}) \frac{y (320-y)}{12800}\\0\end{pmatrix} & \quad \textrm{on } \Gamma_1, \\
#       \eta = 0 & \quad \textrm{on } \Gamma_2, \\
#       u \cdot n = 0 & \quad \textrm{on } \Gamma_3, \\
#
# where :math:`n` is the normal vector pointing outside the domain,
# :math:`\Gamma_1` is the west boundary of the channel, :math:`\Gamma_2` is the
# east boundary of the channel, and :math:`\Gamma_3` is the north and south
# boundaries of the channel.
#
#

# Implementation
# **************
#

# We begin with importing the OpenTidalFarm module.

from opentidalfarm import *
import os

# Create a rectangular domain.
domain = FileDomain("mesh/mesh.xml")

# Specify boundary conditions. For time-dependent boundary condition use
# a parameter named `t` in the :class:`dolfin.Expression` and it will be
# automatically be updated to the current timelevel during the solve.

bcs = BoundaryConditionSet()
u_expr = Expression(("3*sin(pi*t/600.)*x[1]*(320-x[1])*2/(160.*160.)", "0"), t=Constant(0))
bcs.add_bc("u", u_expr, facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)

# Apply a strong no-slip boundary condition. This can be changed to
# free slip (weakly enforced), by leaving out the Constant((0, 0))
# argument and changing bctype to "free_slip"
bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="strong_dirichlet")

# Next we get the default parameters of a shallow water problem and configure it
# to our needs.

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
problem_params.viscosity = Constant(300)
problem_params.depth = Constant(50)
problem_params.g = Constant(9.81)

# Set time parameters
problem_params.start_time = Constant(0)
problem_params.finish_time = Constant(1200)
problem_params.dt = Constant(30)

problem_params.functional_final_time_only = False

# The initial condition consists of three components: u_x, u_y and eta
# Note that we do not set all components to zero, as some components of the
# Jacobian of the quadratic friction term is non-differentiable.
problem_params.initial_condition = Constant((DOLFIN_EPS, 0, 0))

# The next step is to create the turbine farm. In this case, the
# farm consists of only 1 turbine placed in the midle of the channel.

# Before adding the turbine we must specify the type of turbine used in the
# array and what to optimize for.
# Here we used the default BumpTurbine and set the controls to optimize for
# dynamic friction. The diameter and friction are set. The minimum distance
# between turbines if not specified is set to 1.5*diameter.
turbine = BumpTurbine(diameter=20.0, friction=10.0,
                      controls=Controls(dynamic_friction=True))

# A rectangular farm is defined using the domain and the site dimensions.
# The number of time steps must be specifed when optimizing dynamically, but
# the problem_params have a property which calculates it for you.
farm = RectangularFarm(domain, site_x_start=160, site_x_end=480,
                       site_y_start=80, site_y_end=240, turbine=turbine,
                       n_time_steps = problem_params.n_time_steps)

# Turbines are then added to the site in a regular grid layout.
farm.add_regular_turbine_layout(num_x=8, num_y=4)

problem_params.tidal_farm = farm

# Once the parameter have been set, we create the shallow water problem:

problem = SWProblem(problem_params)

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form. Again, we first ask for the default
# parameters, adjust them to our needs and then create the solver object.
# Here we set the solver to output lots of information.

solver_params = CoupledSWSolver.default_parameters()
solver_params.dump_period = 1
solver_params.output_abs_u_at_turbine_positions = True
solver_params.output_j = True
solver_params.output_temporal_breakdown_of_j = True
solver_params.output_control_array = True
solver_params.cache_forward_state = False
solver = CoupledSWSolver(problem, solver_params)

# Next we create a reduced functional, that is the functional considered as a
# pure function of the control by implicitly solving the shallow water
# equations. For
# that we need to specify the objective functional (the value that we want to
# optimize), the control (the variables that we want to change), and our
# shallow water solver.

functional = PowerFunctional(problem)
control = TurbineFarmControl(farm)
rf_params = ReducedFunctionalParameters()
rf_params.automatic_scaling = False
rf = ReducedFunctional(functional, control, solver, rf_params)

# Now we can define the constraints for the controls and start the
# optimisation. (The callback parameter must be set to
# solver.update_optimisation_iteration to get the correct iteration.)

lb, ub = farm.constraints(lower_friction_bounds=0, upper_friction_bounds=1000)
f_opt = maximize(rf, bounds=[lb, ub], method="L-BFGS-B", options={'maxiter':
10, 'ftol':1e-2}, callback=solver.update_optimisation_iteration)

# Reset the scale of the reduced functional, which maximize have changed,
# before calculating the energy output. 

rf.scale = 1.0
solver_params.dump_period = -1
energy = rf(f_opt)

# Finally we print out the result.

print "The opimized friction coefficient for each timestep is: "
print f_opt
print "This gives a energy output of {}.".format(energy)

# How to run the example
# **********************

# The example code can be found in ``examples/channel-dynamic-optimization/`` in the
# ``OpenTidalFarm`` source tree, and executed as follows:

# .. code-block:: bash

#   $ python channel-dynamic-optimization.py
