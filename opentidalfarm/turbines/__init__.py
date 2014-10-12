"""The turbine module encapsulates the specification of the turbines which are
being optimized, how they are controlled and how they are parameterised.

  .. autoclass:: Turbine
  .. autoclass:: TurbineParameterisation
  .. autoclass:: Controls
"""

from .controls import Controls
from .bump_turbine import BumpTurbine
from .smeared_turbine import SmearedTurbine
from .thrust_turbine import ThrustTurbine
from .implicit_thrust_turbine import ImplicitThrustTurbine

__all__ = ["BumpTurbine", "SmearedTurbine", "ThrustTurbine",
           "ImplicitThrustTurbine", "Controls"]
