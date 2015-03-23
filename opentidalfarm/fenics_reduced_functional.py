from dolfin import *
from dolfin_adjoint import *
from solvers import Solver
from functionals import TimeIntegrator, PrototypeFunctional

__all__ = ["FenicsReducedFunctional"]

class FenicsReducedFunctional(object):
    """
    Following parameters are expected:

    :ivar functional: a :class:`PrototypeFunctional` class.
    :ivar controls: a (optionally list of) :class:`dolfin_adjoint.DolfinAdjointControl` object.
    :ivar solver: a :class:`Solver` object.

    This class has a parameter attribute for further adjustments.
    """

    def __init__(self, functional, controls, solver):

        self.solver = solver
        if not isinstance(solver, Solver):
            raise ValueError, "solver argument of wrong type."

        self.functional = functional
        if not isinstance(functional, PrototypeFunctional):
            raise ValueError, "invalid functional argument."

        # Hidden attributes
        self._solver_params = solver.parameters
        self._problem_params = solver.problem.parameters
        self._time_integrator = None

        # Controls
        self.controls = enlisting.enlist(controls)

    def evaluate(self, annotate=True):
        """ Return the functional value for the given control values. """

        log(INFO, 'Start evaluation of j')
        timer = dolfin.Timer("j evaluation")

        farm = self.solver.problem.parameters.tidal_farm

        # Configure dolfin-adjoint
        adj_reset()
        dolfin.parameters["adjoint"]["record_all"] = True

        # Solve the shallow water system and integrate the functional of
        # interest.
        final_only = (not self.solver.problem._is_transient or
                      self._problem_params.functional_final_time_only)
        self.time_integrator = TimeIntegrator(self.solver.problem,
                                              self.functional, final_only)

        for sol in self.solver.solve(annotate=annotate):
            self.time_integrator.add(sol["time"], sol["state"], sol["tf"],
                                     sol["is_final"])

        j = self.time_integrator.integrate()

        timer.stop()

        log(INFO, 'Runtime: %f s.' % timer.value())
        log(INFO, 'j = %e.' % float(j))

        return j

    def derivative(self, forget=False, **kwargs):
        """ Computes the first derivative of the functional with respect to its
        controls by solving the adjoint equations. """

        log(INFO, 'Start evaluation of dj')
        timer = dolfin.Timer("dj evaluation")

        J = self.time_integrator.dolfin_adjoint_functional()
        dj = compute_gradient(J, self.controls, forget=forget, **kwargs)
        dolfin.parameters["adjoint"]["stop_annotating"] = False

        log(INFO, "Runtime: " + str(timer.stop()) + " s")

        return dj
