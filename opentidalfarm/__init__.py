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
from turbine import *
from solvers import *
from problems import *
from domains import *
from functionals import *
from tidal import TidalForcing, BathymetryDepthExpression

from optimisation_helpers import friction_constraints, \
    get_minimum_distance_constraint_func, get_domain_constraints, \
    deploy_turbines, position_constraints, generate_site_constraints, \
    get_distance_function, MinimumDistanceConstraint, \
    PolygonSiteConstraints, DomainRestrictionConstraints
from reduced_functional import ReducedFunctional, ReducedFunctionalParameters
from boundary_conditions import BoundaryConditionSet
from turbine_function import TurbineFunction

# Option management instance.
from options import options

from dolfin import *
from dolfin_adjoint import minimize, maximize, Function, solve

# We set the perturbation_direction with a constant seed, so that it is
# consistent in a parallel environment.
import numpy
numpy.random.seed(21)
