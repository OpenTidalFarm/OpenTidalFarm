from dolfin import Constant
from dolfin_adjoint import Constant
from problem import Problem


class SteadyShallowWaterProblemParameters(object):
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
        be either 'strong_dirichlet' or 'strong_dirichlet'. 
        Default: 'strong_dirichlet'
    :ivar strong_bc: A list of strong boundary conditions. Default: None
    :ivar free_slip_on_sides: Indicates if a free-slip boundary conditions is to
        be applied on the sides. Default: True
    :ivar eta_weak_dirichlet_bc_expr: A list of boundary conditions for the free
        surface. Default: None
    """

    # Physical parameters
    depth = 50.
    g = 9.81
    viscosity = 3.0
    friction = 0.0025

    # Equation settings
    include_advection = True
    include_viscosity = True
    linear_divergence = False

    # Boundary conditions
    bctype = 'strong_dirichlet'
    strong_bc = None
    free_slip_on_sides = True
    eta_weak_dirichlet_bc_expr = None



class SteadyShallowWaterProblem(Problem):

    def __init__(self, parameters):
        self.parameters = parameters

    @property
    def is_transient(self):
        return False

    @staticmethod
    def default_parameters():
        ''' Returns a :class:`SteadyShallowWaterProblemParameters` with default
            parameters. '''

        return SteadyShallowWaterProblemParameters()
