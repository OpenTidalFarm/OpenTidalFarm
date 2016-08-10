"""The turbine module encapsulates the specification of the turbines which are
being optimized, how they are controlled and how they are parameterised.

Turbines
========

Bump turbine
------------

.. automodule:: opentidalfarm.turbines.bump_turbine
    :members:
    :show-inheritance:

Thrust turbine
--------------

.. automodule:: opentidalfarm.turbines.thrust_turbine
    :members:
    :show-inheritance:

.. automodule:: opentidalfarm.turbines.implicit_thrust_turbine
    :members:
    :show-inheritance:

The base turbine
----------------

.. automodule:: opentidalfarm.turbines.base_turbine
    :members:

Controls
--------

.. automodule:: opentidalfarm.turbines.controls
    :members:
    """

from .controls import Controls
from .bump_turbine import BumpTurbine
from .smeared_turbine import SmearedTurbine
from .thrust_turbine import ThrustTurbine
from .implicit_thrust_turbine import ImplicitThrustTurbine

__all__ = ["BumpTurbine", "SmearedTurbine", "ThrustTurbine",
           "ImplicitThrustTurbine", "Controls"]
