"""

The OpenTidalFarm project

"""

__version__ = '0.2'
__author__  = 'Simon Funke'
__credits__ = ['Simon Funke']
__license__ = 'GPL-3'
__maintainer__ = 'Simon Funke'
__email__ = 'simon.funke@gmail.com'

import finite_elements
import helpers
import shallow_water_model
import mini_model 
import initial_conditions
from configuration import *

from optimisation_helpers import friction_constraints, get_minimum_distance_constraint_func, deploy_turbines, position_constraints
from reduced_functional import ReducedFunctional
from dirichlet_bc import DirichletBCSet
from initial_conditions import SinusoidalInitialCondition, BumpInitialCondition
from turbines import Turbines

from dolfin import *
from dolfin_adjoint import minimize, maximize, Function

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
import numpy
numpy.random.seed(21)
