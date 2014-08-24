Getting started 
========================

Following example code shows how to optimise the position of 32 turbines in a mesh of the Orkney islands.

.. literalinclude:: getting_started.py
   :lines: 1-29

This example can be found in the ``examples/tutorial`` directory and can be executed by running ``make mesh && make``.

The output files are

- turbine.pvd: The turbine positions at each optimisation step
- p2p1_u.pvd: The velocity function for the most recent turbine position calculation. 
- p2p1_p.pvd: The free-surface displacement function for the most recent turbine position calculation.

If you only want to compute the power production for the given layout (without optimising), replace the last code line above with:
   
.. literalinclude:: getting_started.py
   :lines: 30-
