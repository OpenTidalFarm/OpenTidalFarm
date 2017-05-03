"""The turbine module encapsulates the specification of the turbines which are
being optimized, how they are controlled and how they are parameterised.

Turbines
========

Bump turbine
------------

.. automodule:: opentidalfarm.turbines.bump_turbine
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
from .bump_turbine import BumpTurbine, standard_thrust_curve
from .smeared_turbine import SmearedTurbine

__all__ = ["BumpTurbine", "SmearedTurbine", "Controls", "standard_thrust_curve"]
