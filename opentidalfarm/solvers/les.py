# Adapted from the Firedrake-Fluids project.
from dolfin import *
from dolfin_adjoint import *

class LES(object):
    r""" A solver for computing the eddy viscosity by solving:

    .. math:: e = (s w)^2I

    where :math:`e` is the eddy viscosity, :math:`s` is the smagorinsky
    coefficient, :math:`w = \sqrt{\text{cell volume}}` is the filter width, and :math:`I` is the second
    invariant defined as:

    .. math:: I = \sum_{1 \le i, j \le 2} 2S_{i, j}^2, \quad S = \frac{1}{2} \left(\nabla u + \nabla u^T\right)

    Parameters:

    :param V: The function space for the the eddy viscosity.
    :param u: The velocity function.
    :param smagorinsky_coefficient: The smagorinsky coefficient.

    :ivar eddy_viscosity: The smagorinsky coefficient.
    """

    def __init__(self, V, u, smagorinsky_coefficient):
        self._V = V
        self.eddy_viscosity = Function(V)

        # Create a eddy viscosity solver
        les_lhs, les_rhs = self._eddy_viscosity_eqn(u, smagorinsky_coefficient)

        eddy_viscosity_problem = LinearVariationalProblem(les_lhs, les_rhs,
                 self.eddy_viscosity, bcs=[])
        self._solver = LinearVariationalSolver(eddy_viscosity_problem)
        self._solver.parameters["linear_solver"] = "lu"
        self._solver.parameters["symmetric"] = True
        self._solver.parameters["lu_solver"]["reuse_factorization"] = True

    def _strain_rate_tensor(self, u):
        S = 0.5*(grad(u) + grad(u).T)
        return S

    def _eddy_viscosity_eqn(self, u, smagorinsky_coefficient):

        dim = len(u)
        w = TestFunction(self._V)
        eddy_viscosity = TrialFunction(self._V)

        cell_vol = CellVolume(self._V.mesh())
        filter_width = cell_vol**(1.0/dim)

        S = self._strain_rate_tensor(u)
        second_invariant = 0.0
        for i in range(0, dim):
           for j in range(0, dim):
              second_invariant += 2.0*(S[i,j]**2)

        second_invariant = sqrt(second_invariant)
        rhs = (smagorinsky_coefficient*filter_width)**2*second_invariant

        lhs = inner(w, eddy_viscosity)*dx
        rhs = inner(w, rhs)*dx

        return lhs, rhs

    def solve(self):
        """ Update the eddy viscosity solution for the current velocity.

        :returns: The eddy viscosity.
        """
        self._solver.solve()
        return self.eddy_viscosity
