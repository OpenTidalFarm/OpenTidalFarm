class Solver(object):
    """A generic implementation of a solver."""

    def __init__(self):
        """ Initialises the solver. """
        self.optimisation_iteration = 0
        self.search_iteration = 0

    def update_optimisation_iteration(self, m):
        self.optimisation_iteration += 1

    @classmethod
    def default_parameters(cls):
        """ Returns a dictionary with the default parameters. """
        return {}

    def solve(self, state, turbine_field, functional=None, annotate=True,
              linear_solver="default", preconditioner="default",
              u_source=None):
        """ Solves the problem """
        pass
