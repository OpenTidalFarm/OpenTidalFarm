"""

The OpenTidalFarm project

"""

__version__ = '2016.2'
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
from tidal import *
from optimisation_helpers import *
from reduced_functional import *
from fenics_reduced_functional import *
from boundary_conditions import *
from turbine_function import *

from dolfin import *
from dolfin import parameters
from dolfin_adjoint import minimize, maximize, Function, solve, Control, \
                           Constant

parameters["form_compiler"]["representation"] = "uflacs"

# We set the perturbation_direction with a constant seed, so that it is
# consistent in a parallel environment.
import numpy
numpy.random.seed(21)
