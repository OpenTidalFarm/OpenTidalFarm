import os.path
from os import mkdir

class Solver(object):
    """A generic implementation of a solver."""

    def __init__(self):
        """ Initialises the solver. """
        self.optimisation_iteration = 0
        self.search_iteration = 0

    def update_optimisation_iteration(self, m):
        self.optimisation_iteration += 1

    def get_optimisation_and_search_directory(self):
        dir = ""
        if hasattr(self, 'output_dir'):
            dir = self.output_dir
        else:
            dir = os.curdir
        dir = os.path.join(dir, "iter_{}".format(self.optimisation_iteration))
        if not os.path.exists(dir):
            mkdir(dir)
        dir = os.path.join(dir, "search_{}".format(self.search_iteration))
        if not os.path.exists(dir):
            mkdir(dir)
        return dir

    @classmethod
    def default_parameters(cls):
        """ Returns a dictionary with the default parameters. """
        return {}

    def solve(self, state, turbine_field, functional=None, annotate=True,
              linear_solver="default", preconditioner="default",
              u_source=None):
        """ Solves the problem """
        pass
