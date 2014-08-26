from types import MethodType
from dolfin import Constant
from dolfin_adjoint import Constant
from problem import Problem
from ..helpers import FrozenClass
from ..boundary_conditions import BoundaryConditionSet
from .. import finite_elements
from ..domains.domain import Domain


class SteadyShallowWaterProblemParameters(FrozenClass):
    """ A set of parameters for a :class:`SteadyShallowWaterProblem`. 
    
    Physical parameters:

    :ivar depth: Water depth. Default: 50
    :ivar g: Gravitational constant. Default: 9.81
    :ivar viscosity: Water viscosity. Default: 3.0
    :ivar friction: Natural bottom friction. Default: 0.0025

    Equation parameters:

    :ivar include_advection: Boolean indicating if the advection is included. 
        Default: True
    :ivar include_viscosity: Boolean indicating if the viscosity is included. 
        Default: True
    :ivar linear_divergence: Boolean indicating if the divergence equation is 
        linearised. Default: False

    Boundary conditions:

    :ivar bctype: Specifies how the boundary conditions should be enforced. Must
        be one of 'dirichlet', 'strong_dirichlet' or 'flather'. 
        Default: 'strong_dirichlet'
    :ivar strong_bc: A list of strong boundary conditions. Default: None
    :ivar flather_bc_expr: A :class:`dolfin.Expression` describing the Flather
        boundary condition values. Default: None
    :ivar u_weak_dirichlet_bc_expr: A :class:`dolfin.Expression` describing the weak
        Dirichlet boundary condition values for the velocity.
        Default: None.
    :ivar eta_weak_dirichlet_bc_expr: A :class:`dolfin.Expression` describing the weak
        Dirichlet boundary condition values for the free surface displacment.
        Default: None.
    :ivar free_slip_on_sides: Indicates if a free-slip boundary conditions is to
        be applied on the sides. Default: True
    """

    # Domain
    domain = None

    # Physical parameters
    depth = 50.
    g = 9.81
    viscosity = 3.0
    friction = 0.0025

    # Equation settings
    include_advection = True
    include_viscosity = True
    linear_divergence = False
    initial_condition = Constant((1e-16, 0, 0))

    # Finite element settings
    finite_element = staticmethod(finite_elements.p2p1)

    # Boundary conditions
    bcs = BoundaryConditionSet()

class SteadyShallowWaterProblem(Problem):

    def __init__(self, parameters):
        """ Instantiates a new :class:`SteadyShallowWaterProblem` object.

            :parameter parameters: A :class:`SteadyShallowWaterProblemParameters`
                object containing the parameters of the problem.
        """

        if not isinstance(parameters, SteadyShallowWaterProblemParameters):
            raise TypeError, "parameters must be of type \
ShallowWaterProblemParameters."

        if not isinstance(parameters.domain, Domain):
            raise TypeError, "parameters.domain is not a valid Domain."

        self.parameters = parameters

    @property
    def _is_transient(self):
        return False

    @staticmethod
    def default_parameters():
        ''' Returns a :class:`SteadyShallowWaterProblemParameters` with default
            parameters. '''

        return SteadyShallowWaterProblemParameters()
