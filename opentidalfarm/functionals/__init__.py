"""
Initialises the various functionals. All functionals should overload the
Prototype, as this allows them to be combined (added, subtracted and scaled).
Functionals should be grouped by type; Power / Cost / Environment etc.
"""

from power_functionals import PowerFunctional
from prototype_functional import PrototypeFunctional
from time_integrator import TimeIntegrator
