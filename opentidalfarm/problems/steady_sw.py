from types import MethodType
from dolfin import Constant
from dolfin_adjoint import Constant
from problem import Problem
from ..helpers import FrozenClass
from ..boundary_conditions import BoundaryConditionSet
from .. import finite_elements
from ..domains.domain import Domain


class SteadySWProblemParameters(FrozenClass):
    """ A parameters set for a :class:`SteadySWProblem`.

    Domain parameters:

    :ivar domain: The computational domain, see :doc:`opentidalfarm.domains`.

    Physical parameters:

    :ivar depth: Water depth. Default: 50.0
    :ivar g: Gravitational constant. Default: 9.81
    :ivar viscosity: Water viscosity. Default: 3.0
    :ivar friction: Natural bottom friction. Default: 0.0025
    :ivar rho: Density of water. Default: 1025.0

    Equation parameters:

    :ivar include_advection: Boolean indicating if the advection is included.
        Default: True
    :ivar include_viscosity: Boolean indicating if the viscosity is included.
        Default: True
    :ivar linear_divergence: Boolean indicating if the divergence equation is
        linearised. Default: False
    :ivar f_u: A source term for the velocity component as a
        :class:`dolfin.Expression`. Default: Constant((0, 0))

    Boundary conditions:

    :ivar bctype: Specifies how the boundary conditions should be enforced.
        Valid options are: 'weak_dirichlet', 'strong_dirichlet' or 'flather'.
        Default: 'strong_dirichlet'
    :ivar bcs: A :class:`opentidalfarm.boundary_conditions.BoundaryConditionSet` containing a list of boundary
        conditions for the problem.

    Discretization settings:

    :ivar finite_element: The finite-element pair to use. Default:
    :class:`opentidalfarm.finite_elements.p2p1`
    """

    # Domain
    domain = None

    # Physical parameters
    depth = 50.
    g = 9.81
    viscosity = 3.0
    friction = 0.0025
    rho = 1025.
    tidal_farm = None

    # Equation settings
    include_advection = True
    include_viscosity = True
    linear_divergence = False
    # FIXME: Replace initial_condition with initial_condition_*
    initial_condition = Constant((1e-16, 0, 0))
    initial_condition_u = Constant((1e-16, 0))
    initial_condition_eta = Constant(0)
    f_u = Constant((0, 0))

    # Finite element settings
    finite_element = staticmethod(finite_elements.p2p1)

    # Boundary conditions
    bcs = BoundaryConditionSet()

class SteadySWProblem(Problem):
    r""" Create a steady-state shallow water problem:

        .. math:: -\nabla\cdot\nu\nabla u+u\cdot\nabla u+g\nabla
            \eta + \frac{c_b + c_t}{H} \| u\| u &= f_u, \\
            \nabla \cdot \left( H u \right) &= 0,

        where

        - :math:`u` is the velocity,
        - :math:`\eta` is the free-surface displacement,
        - :math:`H=\eta + h` is the total water depth where :math:`h` is the
          water depth at rest,
        - :math:`f_u` is the velocity forcing term,
        - :math:`c_b` is the (quadratic) natural bottom friction coefficient,
        - :math:`c_t` is the (quadratic) friction coefficient due to the turbine
          farm,
        - :math:`\nu` is the viscosity coefficient,
        - :math:`g` is the gravitational constant,

        :parameter parameters: A :class:`SteadySWProblemParameters`
            object containing the parameters of the problem.
    """

    def __init__(self, parameters):

        if not type(parameters) == SteadySWProblemParameters:
            raise TypeError("parameters must be of type \
SteadySWProblemParameters.")

        self.__init_without_type_check__(parameters)

    def __init_without_type_check__(self, parameters):

        if not isinstance(parameters.domain, Domain):
            raise TypeError("parameters.domain is not a valid Domain.")

        self.parameters = parameters

    @property
    def _is_transient(self):
        return False

    @staticmethod
    def default_parameters():
        ''' Returns a :class:`SteadySWProblemParameters` with default
            parameters. '''

        return SteadySWProblemParameters()
