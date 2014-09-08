import sys
import os.path
import numpy
import helpers
import dolfin_adjoint
from dolfin import *
from dolfin_adjoint import *
from turbines import *
from solvers import Solver
from functionals import FunctionalIntegrator, FunctionalPrototype
from memoize import MemoizeMutable


class ReducedFunctionalParameters(helpers.FrozenClass):
    """ A set of parameters for a :class:`ReducedFunctional`. 

    Following parameters are available:

    :ivar scale: A scaling factor. Default: 1.0
    :ivar automatic_scaling: The reduced functional will be
        automatically scaled such that the maximum absolute value of the initial
        gradient is equal to the specified factor. Set to False to deactivate the 
        automatic scaling. Default: 5.
    :ivar save_checkpoints: Automatically store checkpoints after each
        optimisation iteration. Default: False
    """

    scale = 1.
    automatic_scaling = 5.
    save_checkpoints = False


class ReducedFunctionalNumPy(dolfin_adjoint.ReducedFunctionalNumPy):
    """ 
    Following parameters are expected:

    :ivar functional: a :class:`FunctionalPrototype` class.
    :ivar solver: a :class:`Solver` object.
    :ivar parameters: a :class:`ReducedFunctionalParameters` object.

    This class has a parameter attribute for further adjustments.
    """

    def __init__(self, functional, solver, parameters):

        # For consistency with the dolfin-adjoint API
        self.scale = parameters.scale

        self.solver = solver
        if not isinstance(solver, Solver):
            raise ValueError, "solver argument of wrong type."

        self.functional = functional
        #if not isinstance(functional, FunctionalPrototype):
        #    raise ValueError, "invalid functional argument."

        self._farm = solver.problem.parameters.tidal_farm
        if self._farm is None:
            raise ValueError, "The problem must have a tidal farm."

        # Create the default parameters
        self.parameters = parameters

        # Hidden attributes
        self._solver_params = solver.parameters
        self._problem_params = solver.problem.parameters
        self._integrator = None
        self._optimisation_iteration = 0
        self._automatic_scaling_factor = None

        # Caching variables that store which controls the last forward run was performed
        self.last_m = None
        if self.solver.parameters.dump_period > 0:
            turbine_filename = solver_parameters.output_dir + os.path.sep + \
                               "turbines.pvd"
            self.turbine_file = File(turbine_filename, "compressed")

            if self._solver_params.output_turbine_power:
                power_filename = solver_parameters.output_dir + os.path.sep + \
                                 "power.pvd"
                self.power_file = File(power_filename, "compressed")

        self.parameter = [TurbineFarmParameter(self._farm)]

        # For smeared turbine parametrisations we only want to store the 
        # hash of the control values into the pickle datastructure
        use_hash_keys = (self._farm.params["turbine_parametrisation"] == "smeared")

        self.compute_functional_mem = MemoizeMutable(self.compute_functional, 
            use_hash_keys)
        self.compute_gradient_mem = MemoizeMutable(self.compute_gradient, 
            use_hash_keys)

    def compute_gradient(self, m, forget=True):
        ''' Compute the functional gradient for the turbine positions/frictions array '''

        # If any of the parameters changed, the forward model needs to re-run
        if numpy.any(m != self.last_m):
            self.compute_functional(m, annotate=True)

        final_only = not self.solver.problem._is_transient or \
                     self._problem_params.functional_final_time_only
        functional = self.functional(self._farm, rho=self._problem_params.rho)
        integrator = FunctionalIntegrator(self.solver.problem, functional,
                                          final_only)

        J = self.integrator.dolfin_adjoint_functional()

        # Output power
        if self.solver.parameters.dump_period > 0:
            if self._farm.params['output_turbine_power']:
                turbines = self._farm.turbine_cache.cache["turbine_field"]
                power = self.functional(self._farm).power(solver.current_state, turbines)
                self.power_file << project(power,
                                           self._farm.turbine_function_space, 
                                           annotate=False)

        if 'dynamic_turbine_friction' in self._farm.params["controls"]:
            parameters = [FunctionControl("turbine_friction_cache_t_%i" % i)
                    for i in range(len(self._farm.params["turbine_friction"]))]

        else:
            parameters = FunctionControl("turbine_friction_cache")

        djdtf = dolfin_adjoint.compute_gradient(J, parameters, forget=forget)
        dolfin.parameters["adjoint"]["stop_annotating"] = False

        # Decide if we need to apply the chain rule to get the gradient of interest
        if self._farm.params['turbine_parametrisation'] == 'smeared':
            # We are looking for the gradient with respect to the friction
            dj = dolfin_adjoint.optimization.get_global(djdtf)

        else:
            # Let J be the functional, m the parameter and u the solution of the PDE equation F(u) = 0.
            # Then we have
            # dJ/dm = (\partial J)/(\partial u) * (d u) / d m + \partial J / \partial m
            #               = adj_state * \partial F / \partial u + \partial J / \partial m
            # In this particular case m = turbine_friction, J = \sum_t(ft)
            dj = []

            if 'turbine_friction' in self._farm.params["controls"]:
                # Compute the derivatives with respect to the turbine friction
                for tfd in self._farm.turbine_cache.cache["turbine_derivative_friction"]:
                    self._farm.turbine_cache.update(self._farm)
                    dj.append(djdtf.vector().inner(tfd.vector()))

            elif 'dynamic_turbine_friction' in self._farm.params["controls"]:
                # Compute the derivatives with respect to the turbine friction
                for djdtf_arr, t in zip(djdtf, self._farm.turbine_cache.cache["turbine_derivative_friction"]):
                    for tfd in t:
                        self._farm.turbine_cache.update(self._farm)
                        dj.append(djdtf_arr.vector().inner(tfd.vector()))

            if 'turbine_pos' in self._farm.params["controls"]:
                # Compute the derivatives with respect to the turbine position
                for d in self._farm.turbine_cache.cache["turbine_derivative_pos"]:
                    for var in ('turbine_pos_x', 'turbine_pos_y'):
                        self._farm.turbine_cache.update(self._farm)
                        tfd = d[var]
                        dj.append(djdtf.vector().inner(tfd.vector()))

            dj = numpy.array(dj)

        return dj

    def compute_functional(self, m, annotate=True):
        ''' Compute the functional of interest for the turbine positions/frictions array '''

        self.last_m = m

        self.update_turbine_cache(m)
        tf = self._farm.turbine_cache.cache["turbine_field"]

        return self._compute_functional_from_tf(tf, annotate=annotate)

    def _compute_functional_from_tf(self, tf, annotate=True):
        ''' Computes the functional of interest for a given turbine friction function. '''

        # Reset the dolfin-adjoint tape
        adj_reset()

        dolfin.parameters["adjoint"]["record_all"] = True
        self._set_revolve_parameters()

        # Solve the shallow water system and integrate the functional of
        # interest.
        final_only = not self.solver.problem._is_transient or \
                     self._problem_params.functional_final_time_only
        functional = self.functional(self._farm, rho=self._problem_params.rho)
        self.integrator = FunctionalIntegrator(self.solver.problem, 
                                               functional, 
                                               final_only)

        for sol in self.solver.solve(turbine_field=tf, 
                                annotate=annotate):
            self.integrator.add(sol["time"], sol["state"], sol["tf"], 
                           sol["is_final"])

        return self.integrator.integrate()

    def _set_revolve_parameters(self):
        if (hasattr(self._solver_params, "revolve_parameters") and 
            self._solver_params.revolve_parameters is not None):
          (strategy, snaps_on_disk, snaps_in_ram, verbose) = self._farm.params['revolve_parameters']
          adj_checkpointing(strategy, 
                  self._problem_params.finish_time / self._problem_params.dt,
                  snaps_on_disk=snaps_on_disk, snaps_in_ram=snaps_in_ram, 
                  verbose=verbose)

    def update_turbine_cache(self, m):
        ''' Reconstructs the parameters from the flattened parameter array m and updates the configuration. '''

        if self._farm.params["turbine_parametrisation"] == "smeared":
            self._farm.params["turbine_friction"] = m

        else:
            shift = 0
            if 'turbine_friction' in self._farm.params['controls']:
                shift = len(self._farm.params["turbine_friction"])
                self._farm.params["turbine_friction"] = m[:shift]

            elif 'dynamic_turbine_friction' in self._farm.params['controls']:
                shift = len(numpy.reshape(self._farm.params["turbine_friction"], -1))
                nb_turbines = len(self._farm.params["turbine_pos"])
                self._farm.params["turbine_friction"] = numpy.reshape(m[:shift], (-1, nb_turbines)).tolist()

            if 'turbine_pos' in self._farm.params['controls']:
                mp = m[shift:]
                self._farm.params["turbine_pos"] = numpy.reshape(mp, (-1, 2)).tolist()

        # Set up the turbine field
        self._farm.turbine_cache.update(self._farm)

    def save_checkpoint(self, base_filename):
        ''' Checkpoint the reduceduced functional from which can be used to restart the turbine optimisation. '''
        base_path = os.path.join(self._farm.params["base_path"], base_filename)
        self.compute_functional_mem.save_checkpoint(base_path + "_fwd.dat")
        self.compute_gradient_mem.save_checkpoint(base_path + "_adj.dat")

    def load_checkpoint(self, base_filename='checkpoint'):
        ''' Checkpoint the reduceduced functional from which can be used to restart the turbine optimisation. '''
        base_path = os.path.join(self._farm.params["base_path"], base_filename)
        self.compute_functional_mem.load_checkpoint(base_path + "_fwd.dat")
        self.compute_gradient_mem.load_checkpoint(base_path + "_adj.dat")

    def j(self, m, annotate=True):
        ''' This memoised function returns the functional value for the parameter choice m. '''
        log(INFO, 'Start evaluation of j')
        timer = dolfin.Timer("j evaluation")
        j = self.compute_functional_mem(m, annotate=annotate)
        timer.stop()

        if self.parameters.save_checkpoints:
            self.save_checkpoint("checkpoint")

        log(INFO, 'Runtime: %f s.' % timer.value())
        log(INFO, 'j = %e.' % float(j))
        self.last_j = j

        if self.parameters.automatic_scaling:
            if self._automatic_scaling_factor is None:
                # Computing dj will set the automatic scaling factor.
                log(INFO, "Computing derivative to determine the automatic scaling factor")
                self.dj(m, forget=False, optimisation_iteration=False)
            return j * self.scale * self._automatic_scaling_factor
        else:
            return j * self.scale

    def dj(self, m, forget, optimisation_iteration=True):
        ''' This memoised function returns the gradient of the functional for the parameter choice m. '''
        log(INFO, 'Start evaluation of dj')
        timer = dolfin.Timer("dj evaluation")
        dj = self.compute_gradient_mem(m, forget)

        # We assume that the gradient is computed at and only at the beginning of each new optimisation iteration.
        # Hence, this is the right moment to store the turbine friction field and to increment the optimisation iteration
        # counter.
        if optimisation_iteration:
            self._optimisation_iteration += 1
            if self.solver.parameters.dump_period > 0:
                # A cache hit skips the turbine cache update, so we need
                # trigger it manually.
                if self.compute_gradient_mem.has_cache(m, forget):
                    self.update_turbine_cache(m)
                if "dynamic_turbine_friction" in self._farm.params["controls"]:
                    log(WARNING, "Turbine VTU output not yet implemented for dynamic turbine control")
                else:
                    self.turbine_file << self._farm.turbine_cache.cache["turbine_field"]
                    # Compute the total amount of friction due to turbines
                    if self._farm.params["turbine_parametrisation"] == "smeared":
                        print "Total amount of friction: ", assemble(self._farm.turbine_cache.cache["turbine_field"] * dx)

        if self.parameters.save_checkpoints:
            self.save_checkpoint("checkpoint")

        log(INFO, 'Runtime: ' + str(timer.stop()) + " s")
        log(INFO, '|dj| = ' + str(numpy.linalg.norm(dj)))

        if self.parameters.automatic_scaling:
            self._set_automatic_scaling_factor(dj)
            return dj * self.scale * self._automatic_scaling_factor
        else:
            return dj * self.scale

    def _set_automatic_scaling_factor(self, dj):
        """ Compute the scaling factor if never done before. """

        if self._automatic_scaling_factor is None:
            if not 'turbine_pos' in self._farm.params['controls']:
                raise NotImplementedError("Automatic scaling only works if the turbine positions are control parameters")

            if len(self._farm.params['controls']) > 1:
                assert(len(dj) % 3 == 0)
                # Exclude the first third from the automatic scaling as it contains the friction coefficients
                djl2 = max(abs(dj[len(dj) / 3:]))
            else:
                djl2 = max(abs(dj))

            if djl2 == 0:
                log(ERROR, "Automatic scaling failed: The gradient at the parameter point is zero.")
            else:
                self._automatic_scaling_factor = abs(self.parameters.automatic_scaling *
                        max(self._farm.params['turbine_x'],
                            self._farm.params['turbine_y']) / djl2 /
                        self.scale)
                log(INFO, "Set automatic scaling factor to %e." % self._automatic_scaling_factor)

    def dj_with_check(self, m, seed=0.1, tol=1.8, forget=True):
        ''' This function checks the correctness and returns the gradient of the functional for the parameter choice m. '''

        log(INFO, "Checking derivative at m = " + str(m))
        p = numpy.random.rand(len(m))
        minconv = helpers.test_gradient_array(self.j, self.dj, m, seed=seed, perturbation_direction=p)
        if minconv < tol:
            log(ERROR, "The gradient taylor remainder test failed.")
            sys.exit(1)
        else:
            log(INFO, "The gradient taylor remainder test passed.")

        return self.dj(m, forget)

    def initial_control(self):
        ''' This function returns the control variable array that derives from the initial configuration. '''
        res = []
        if self._farm.params["turbine_parametrisation"] == "smeared":
            res = numpy.zeros(self._farm.turbine_function_space.dim())

        else:
            if ('turbine_friction' in self._farm.params["controls"] or
                'dynamic_turbine_friction' in self._farm.params["controls"]):
                res += numpy.reshape(self._farm.params['turbine_friction'], -1).tolist()

            if 'turbine_pos' in self._farm.params["controls"]:
                res += numpy.reshape(self._farm.params['turbine_pos'], -1).tolist()

        return numpy.array(res)

    def __call__(self, m):
        ''' Interface function for dolfin_adjoint.ReducedFunctional '''

        return self.j(m)

    def derivative(self, m_array, taylor_test=False, seed=0.001, forget=True, **kwargs):
        ''' Interface function for dolfin_adjoint.ReducedFunctional '''

        if taylor_test:
            return self.dj_with_check(m_array, seed, forget)
        else:
            return self.dj(m_array, forget)

    def obj_to_array(self, obj):

        return dolfin_adjoint.optimization.get_global(obj)

    def set_parameters(self, m_array):

        m = [p.data() for p in self.parameter]
        dolfin_adjoint.optimization.set_local(m, m_array)


class ReducedFunctional(ReducedFunctionalNumPy):
    pass


class TurbineFarmVariable(object):
    name = ""

class TurbineFarmParameter(object):
    var = TurbineFarmVariable()

    def __init__(self, farm):
        self._farm = farm

    def data(self):
        if self._farm.params["turbine_parametrisation"] == "smeared":
            m = self._smeared_data()
        else:
            m = self._discrete_data()
        return numpy.array(m)

    def _smeared_data(self):
        if len(self._farm.params["turbine_friction"]) == 0:
            # If the user has not set the turbine friction it is initialised here
            return numpy.zeros(self._farm.turbine_function_space.dim())
        else:
            return self._farm.params["turbine_friction"]

    def _discrete_data(self):
        m = []
        if 'turbine_friction' in self._farm.params["controls"]:
            m += list(self._farm.params['turbine_friction'])
        if 'turbine_pos' in self._farm.params["controls"]:
            m += numpy.reshape(self._farm.params['turbine_pos'], -1).tolist()
        return m

