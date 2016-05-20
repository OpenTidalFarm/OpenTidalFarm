from firedrake import *
from firedrake_adjoint import *
from solvers import Solver
from functionals import TimeIntegrator, PrototypeFunctional

__all__ = ["FenicsReducedFunctional"]

class FenicsReducedFunctional(ReducedFunctional):
    """
    Following parameters are expected:

    :ivar functional: a :class:`PrototypeFunctional` class.
    :ivar controls: a (optionally list of) :class:`firedrake_adjoint.DolfinAdjointControl` object.
    :ivar solver: a :class:`Solver` object.

    This class has a parameter attribute for further adjustments.
    """

    def __init__(self, functional, controls, solver):

        self.solver = solver
        if not isinstance(solver, Solver):
            raise ValueError, "solver argument of wrong type."

        self._functional = functional
        if not isinstance(functional, PrototypeFunctional):
            raise ValueError, "invalid functional argument."

        # Hidden attributes
        self._solver_params = solver.parameters
        self._problem_params = solver.problem.parameters
        self._time_integrator = None

        # Controls
        self.controls = enlisting.enlist(controls)

        # Conform to firedrake-adjoint API
        self.scale = 1
        self.eval_cb_pre = lambda *args: None
        self.eval_cb_post = lambda *args: None
        self.derivative_cb_pre = lambda *args: None
        self.derivative_cb_post = lambda *args: None
        self.replay_cb = lambda *args: None
        self.hessian_cb = lambda *args: None
        self.cache = None
        self.current_func_value = None
        self.hessian = None
        self.functional = Functional(None)


    def evaluate(self, annotate=True):
        """ Computes the functional value by running the forward model. """

        log(INFO, 'Start evaluation of j')
        timer = firedrake.Timer("j evaluation")

        farm = self.solver.problem.parameters.tidal_farm

        # Configure firedrake-adjoint
        adj_reset()
        firedrake.parameters["adjoint"]["record_all"] = True

        # Solve the shallow water system and integrate the functional of
        # interest.
        final_only = (not self.solver.problem._is_transient or
                      self._problem_params.functional_final_time_only)
        self.time_integrator = TimeIntegrator(self.solver.problem,
                                              self._functional, final_only)

        for sol in self.solver.solve(annotate=annotate):
            self.time_integrator.add(sol["time"], sol["state"], sol["tf"],
                                     sol["is_final"])

        j = self.time_integrator.integrate()

        timer.stop()

        log(INFO, 'Runtime: %f s.' % timer.elapsed()[0])
        log(INFO, 'j = %e.' % float(j))

        return j


    def __call__(self, value):
        """ Evaluates the reduced functional for the given control value.

        Args:
            value: The point in control space where to perform the Taylor test. Must be of the same type as the Control (e.g. Function, Constant or lists of latter).

        Returns:
            float: The functional value.
        """

        # Update the control values.
        # Note that we do not update the control values on the tape,
        # because OpenTidalFarm reannotates the tape in each iteartion.
	for c, v in zip(self.controls, value):
            vec = c.coeff.vector()
            vec.zero()
            vec.axpy(1, v.vector())

        return self.evaluate()


    def derivative(self, forget=False, **kwargs):
        """ Computes the first derivative of the functional with respect to its
        controls by solving the adjoint equations. """

        log(INFO, 'Start evaluation of dj')
        timer = firedrake.Timer("dj evaluation")

        J = self.time_integrator.firedrake_adjoint_functional(self.solver.state)
        dj = compute_gradient(J, self.controls, forget=forget, **kwargs)
        firedrake.parameters["adjoint"]["stop_annotating"] = False

        log(INFO, "Runtime: " + str(timer.stop()) + " s")

        return dj
