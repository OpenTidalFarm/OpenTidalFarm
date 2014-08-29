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
import mini_model
import initial_conditions
from configuration import *

from solvers import *
from problems import *
from domains import *

from optimisation_helpers import friction_constraints, get_minimum_distance_constraint_func, get_domain_constraints, deploy_turbines, position_constraints, generate_site_constraints, plot_site_constraints, get_distance_function, MinimumDistanceConstraint, PolygonSiteConstraints, DomainRestrictionConstraints
from reduced_functional import ReducedFunctional
from boundary_conditions import BoundaryConditionSet
from initial_conditions import SinusoidalInitialCondition, BumpInitialCondition
from turbines import Turbines
from functionals import DefaultFunctional, PowerCurveFunctional
from tidal import TidalForcing, BathymetryDepthExpression 

from dolfin import *
from dolfin_adjoint import minimize, maximize, Function, solve
from helpers import info_green, info_red, info_blue, info, print0

from integrator import FunctionalIntegrator

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
import numpy
numpy.random.seed(21)
