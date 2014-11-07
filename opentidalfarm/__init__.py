"""

The OpenTidalFarm project

"""

__version__ = '1.0'
__author__  = 'Simon Funke'
__credits__ = ['Simon Funke']
__license__ = 'GPL-3'
__maintainer__ = 'Simon Funke'
__email__ = 'simon.funke@gmail.com'

import finite_elements
import helpers

from farm import *
from turbines import *
from solvers import *
from problems import *
from domains import *
from functionals import *
from tidal import TidalForcing, BathymetryDepthExpression

from optimisation_helpers import friction_constraints, \
    get_domain_constraints, \
    position_constraints, \
    get_distance_function, \
    ConvexPolygonSiteConstraint, DomainRestrictionConstraints
from reduced_functional import ReducedFunctional, ReducedFunctionalParameters, \
        TurbineFarmControl
from fenics_reduced_functional import FenicsReducedFunctional
from boundary_conditions import BoundaryConditionSet
from turbine_function import TurbineFunction

# Option management instance.
from options import options

from dolfin import *
from dolfin_adjoint import minimize, maximize, Function, solve, Control, \
    Constant

# We set the perturbation_direction with a constant seed, so that it is
# consistent in a parallel environment.
import numpy
numpy.random.seed(21)
