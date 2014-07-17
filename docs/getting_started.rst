Getting started 
========================

Following example code shows how to optimise the position of 32 turbines in a mesh of the Orkney islands.

```python
from opentidalfarm import *

config = SteadyConfiguration("mesh/earth_orkney_converted.xml", inflow_direction=[0.9865837220518425, -0.16325611591095968])
config.params['diffusion_coef'] = 90.0
config.params['turbine_x'] = 40.
config.params['turbine_y'] = 40.
config.params['controls'] = ['turbine_pos']

# Some domain information extracted from the geo file.
# This information is used to deploy the turbines autmatically.
site_x = 1000.
site_y = 500.
site_x_start = 1.03068e+07
site_y_start = 6.52246e+06 - site_y
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place 32 turbines in a regular grid, each with a maximum friction coefficient of 10.5
deploy_turbines(config, nx=8, ny=4, friction=10.5)

# Define some constraints for the optimisation positions.
# Constraint to keep the turbines within the site area 
lb, ub = position_constraints(config)
# Constraint to keep a minium distance of 1.5 turbine diameter between each turbine
ineq = get_minimum_distance_constraint_func(config)

# Solve the optimisation problem
rf = ReducedFunctional(config, plot=True)
maximize(rf, bounds=[lb, ub], constraints=ineq, method="SLSQP")
```

This example can be found in the ``examples/tutorial`` directory and can be executed by running ``make mesh && make``.

The output files are:
* turbine.pvd: The turbine positions at each optimisation step
* p2p1_u.pvd: The velocity function for the most recent turbine position calculation. 
* p2p1_p.pvd: The free-surface displacement function for the most recent turbine position calculation.

If you only want to compute the power production for the given layout (without optimising), replace the ast code line above with:
```python
# Switch off the computation of the automatic scaling factor (requires one adjoint solve), as it is needed only for the optimisation
config.params['automatic_scaling'] = False 

# Retrieve the initial control values (here: turbine positions) 
m0 = rf.initial_control()  

# Store the turbine positions as a pvd file 
rf.update_turbine_cache(m0)
File("turbines.pvd") << config.turbine_cache.cache["turbine_field"]

# Compute the extracted power from the flow
print "Power extraction: %f MW" % rf.j(m0)
```
