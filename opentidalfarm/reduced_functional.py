import numpy
import memoize
import helpers
import sys
import dolfin_adjoint
from dolfin import *
from dolfin_adjoint import *
from turbines import *
from solvers import Solver
from functionals import FunctionalIntegrator, FunctionalPrototype
import os.path


class ReducedFunctionalNumPy(dolfin_adjoint.ReducedFunctionalNumPy):
    """ 

    Following parameters are available:

    :ivar config: FIXME: should be renamed to Control.
    :ivar functional: a :class:`FunctionalPrototype` class.
    :ivar solver: a :class:`Solver` object.
    :ivar scale: an optional scaling factor. Default: 1.0
    :ivar automatic_scaling: if not False, the reduced functional will be
        automatically scaled such that the maximum absolute value of the initial
        gradient is equal to the specified factor. Default: 5.

    """

    def __init__(self, config, functional, solver, scale=1.0, 
                 automatic_scaling=5):
        ''' scale is ignored if automatic_scaling is active. '''
        # Hide the configuration since changes would break the memoize algorithm.

        if not isinstance(solver, Solver):
            raise ValueError, "solver argument of wrong type."

        #if not isinstance(functional, FunctionalPrototype):
        #    raise ValueError, "invalid functional argument."

        self.__config__ = config
        self.scale = scale
        self.solver = solver
        self.automatic_scaling = automatic_scaling
        self.automatic_scaling_factor = None
        self.integrator = None
        self.functional = functional

        # Caching variables that store which controls the last forward run was performed
        self.last_m = None
        self.in_euclidian_space = False  # FIXME: legacy dolfin-adjoint parameter
        if self.solver.parameters.dump_period > 0:
            self.turbine_file = File(config.params['base_path'] + os.path.sep + "turbines.pvd", "compressed")

            if config.params['output_turbine_power']:
                self.power_file = File(config.params['base_path'] + os.path.sep + "power.pvd", "compressed")

        class Variable:
            name = ""

        class Parameter:
            var = Variable()

            def data(self):
                m = []
                if config.params["turbine_parametrisation"] == "smeared":
                    if len(config.params["turbine_friction"]) == 0:
                        # If the user has not set the turbine friction it is initialised here
                        m = numpy.zeros(config.turbine_function_space.dim())
                    else:
                        m = config.params["turbine_friction"]
                else:
                    if 'turbine_friction' in config.params["controls"]:
                        m += list(config.params['turbine_friction'])
                    if 'turbine_pos' in config.params["controls"]:
                        m += numpy.reshape(config.params['turbine_pos'], -1).tolist()
                return numpy.array(m)

        self.parameter = [Parameter()]

        def compute_functional(m, annotate=True):
            ''' Compute the functional of interest for the turbine positions/frictions array '''

            self.last_m = m

            self.update_turbine_cache(m)
            tf = config.turbine_cache.cache["turbine_field"]

            return compute_functional_from_tf(tf, annotate=annotate)

        def compute_functional_from_tf(tf, annotate=True):
            ''' Computes the functional of interest for a given turbine friction function. '''

            # Reset the dolfin-adjoint tape
            adj_reset()

            parameters["adjoint"]["record_all"] = True
            if config.params['revolve_parameters'] is not None:
              (strategy, snaps_on_disk, snaps_in_ram, verbose) = config.params['revolve_parameters']
              adj_checkpointing(strategy, 
                      solver.problem.parameters.finish_time / solver.problem.parameters.dt,
                      snaps_on_disk=snaps_on_disk, snaps_in_ram=snaps_in_ram, 
                      verbose=verbose)

            # Solve the shallow water system and integrate the functional of
            # interest.
            final_only = not solver.problem._is_transient or \
                         solver.problem.parameters.functional_final_time_only
            functional = self.functional(config, rho=solver.problem.parameters.rho)
            self.integrator = FunctionalIntegrator(solver.problem, 
                                                   functional, 
                                                   final_only)

            for sol in solver.solve(turbine_field=tf, 
                                    annotate=annotate):
                self.integrator.add(sol["time"], sol["state"], sol["tf"], 
                               sol["is_final"])

            return self.integrator.integrate()

        def compute_gradient(m, forget=True):
            ''' Compute the functional gradient for the turbine positions/frictions array '''

            # If any of the parameters changed, the forward model needs to re-run
            if numpy.any(m != self.last_m):
                compute_functional(m, annotate=True)

            final_only = not solver.problem._is_transient or \
                         solver.problem.parameters.functional_final_time_only
            functional = self.functional(config, rho=solver.problem.parameters.rho)
            integrator = FunctionalIntegrator(solver.problem, functional,
                                              final_only)

            J = self.integrator.dolfin_adjoint_functional()

            # Output power
            if self.solver.parameters.dump_period > 0:
                if config.params['output_turbine_power']:
                    turbines = self.__config__.turbine_cache.cache["turbine_field"]
                    power = self.functional(config).power(solver.current_state, turbines)
                    self.power_file << project(power,
                                               config.turbine_function_space, 
                                               annotate=False)

            if 'dynamic_turbine_friction' in config.params["controls"]:
                parameters = [FunctionControl("turbine_friction_cache_t_%i" % i) for i in range(len(config.params["turbine_friction"]))]

            else:
                parameters = FunctionControl("turbine_friction_cache")

            djdtf = dolfin_adjoint.compute_gradient(J, parameters, forget=forget)
            dolfin.parameters["adjoint"]["stop_annotating"] = False

            # Decide if we need to apply the chain rule to get the gradient of interest
            if config.params['turbine_parametrisation'] == 'smeared':
                # We are looking for the gradient with respect to the friction
                dj = dolfin_adjoint.optimization.get_global(djdtf)

            else:
                # Let J be the functional, m the parameter and u the solution of the PDE equation F(u) = 0.
                # Then we have
                # dJ/dm = (\partial J)/(\partial u) * (d u) / d m + \partial J / \partial m
                #               = adj_state * \partial F / \partial u + \partial J / \partial m
                # In this particular case m = turbine_friction, J = \sum_t(ft)
                dj = []

                if 'turbine_friction' in config.params["controls"]:
                    # Compute the derivatives with respect to the turbine friction
                    for tfd in config.turbine_cache.cache["turbine_derivative_friction"]:
                        config.turbine_cache.update(config)
                        dj.append(djdtf.vector().inner(tfd.vector()))

                elif 'dynamic_turbine_friction' in config.params["controls"]:
                    # Compute the derivatives with respect to the turbine friction
                    for djdtf_arr, t in zip(djdtf, config.turbine_cache.cache["turbine_derivative_friction"]):
                        for tfd in t:
                            config.turbine_cache.update(config)
                            dj.append(djdtf_arr.vector().inner(tfd.vector()))

                if 'turbine_pos' in config.params["controls"]:
                    # Compute the derivatives with respect to the turbine position
                    for d in config.turbine_cache.cache["turbine_derivative_pos"]:
                        for var in ('turbine_pos_x', 'turbine_pos_y'):
                            config.turbine_cache.update(config)
                            tfd = d[var]
                            dj.append(djdtf.vector().inner(tfd.vector()))

                dj = numpy.array(dj)

            return dj

        # For smeared turbine parametrisations we only want to store the 
        # hash of the control values into the pickle datastructure
        hash_keys = (config.params["turbine_parametrisation"] == "smeared")

        self.compute_functional_mem = memoize.MemoizeMutable(compute_functional, hash_keys)
        self.compute_gradient_mem = memoize.MemoizeMutable(compute_gradient, hash_keys)

    def update_turbine_cache(self, m):
        ''' Reconstructs the parameters from the flattened parameter array m and updates the configuration. '''

        if self.__config__.params["turbine_parametrisation"] == "smeared":
            self.__config__.params["turbine_friction"] = m

        else:
            shift = 0
            if 'turbine_friction' in self.__config__.params['controls']:
                shift = len(self.__config__.params["turbine_friction"])
                self.__config__.params["turbine_friction"] = m[:shift]

            elif 'dynamic_turbine_friction' in self.__config__.params['controls']:
                shift = len(numpy.reshape(self.__config__.params["turbine_friction"], -1))
                nb_turbines = len(self.__config__.params["turbine_pos"])
                self.__config__.params["turbine_friction"] = numpy.reshape(m[:shift], (-1, nb_turbines)).tolist()

            if 'turbine_pos' in self.__config__.params['controls']:
                mp = m[shift:]
                self.__config__.params["turbine_pos"] = numpy.reshape(mp, (-1, 2)).tolist()

        # Set up the turbine field
        self.__config__.turbine_cache.update(self.__config__)

    def save_checkpoint(self, base_filename):
        ''' Checkpoint the reduceduced functional from which can be used to restart the turbine optimisation. '''
        base_path = os.path.join(self.__config__.params["base_path"], base_filename)
        self.compute_functional_mem.save_checkpoint(base_path + "_fwd.dat")
        self.compute_gradient_mem.save_checkpoint(base_path + "_adj.dat")

    def load_checkpoint(self, base_filename='checkpoint'):
        ''' Checkpoint the reduceduced functional from which can be used to restart the turbine optimisation. '''
        base_path = os.path.join(self.__config__.params["base_path"], base_filename)
        self.compute_functional_mem.load_checkpoint(base_path + "_fwd.dat")
        self.compute_gradient_mem.load_checkpoint(base_path + "_adj.dat")

    def j(self, m, annotate=True):
        ''' This memoised function returns the functional value for the parameter choice m. '''
        log(INFO, 'Start evaluation of j')
        timer = dolfin.Timer("j evaluation")
        j = self.compute_functional_mem(m, annotate=annotate)
        timer.stop()

        if self.__config__.params["save_checkpoints"]:
            self.save_checkpoint("checkpoint")

        log(INFO, 'Runtime: %f s.' % timer.value())
        log(INFO, 'j = %e.' % float(j))
        self.last_j = j

        if self.automatic_scaling:
            if self.automatic_scaling_factor is None:
                # Computing dj will set the automatic scaling factor.
                log(INFO, "Computing derivative to determine the automatic scaling factor")
                self.dj(m, forget=False, optimisation_iteration=False)
            return j * self.scale * self.automatic_scaling_factor
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
            self.__config__.optimisation_iteration += 1
            if self.solver.parameters.dump_period > 0:
                # A cache hit skips the turbine cache update, so we need
                # trigger it manually.
                if self.compute_gradient_mem.has_cache(m, forget):
                    self.update_turbine_cache(m)
                if "dynamic_turbine_friction" in self.__config__.params["controls"]:
                    log(WARNING, "Turbine VTU output not yet implemented for dynamic turbine control")
                else:
                    self.turbine_file << self.__config__.turbine_cache.cache["turbine_field"]
                    # Compute the total amount of friction due to turbines
                    if self.__config__.params["turbine_parametrisation"] == "smeared":
                        print "Total amount of friction: ", assemble(self.__config__.turbine_cache.cache["turbine_field"] * dx)

        if self.__config__.params["save_checkpoints"]:
            self.save_checkpoint("checkpoint")

        log(INFO, 'Runtime: ' + str(timer.stop()) + " s")
        log(INFO, '|dj| = ' + str(numpy.linalg.norm(dj)))

        if self.automatic_scaling:
            self._set_automatic_scaling_factor(dj)
            return dj * self.scale * self.automatic_scaling_factor
        else:
            return dj * self.scale

    def _set_automatic_scaling_factor(self, dj):
        """ Compute the scaling factor if never done before. """

        if self.automatic_scaling_factor is None:
            if not 'turbine_pos' in self.__config__.params['controls']:
                raise NotImplementedError("Automatic scaling only works if the turbine positions are control parameters")

            if len(self.__config__.params['controls']) > 1:
                assert(len(dj) % 3 == 0)
                # Exclude the first third from the automatic scaling as it contains the friction coefficients
                djl2 = max(abs(dj[len(dj) / 3:]))
            else:
                djl2 = max(abs(dj))

            if djl2 == 0:
                log(ERROR, "Automatic scaling failed: The gradient at the parameter point is zero.")
            else:
                self.automatic_scaling_factor = abs(self.automatic_scaling * max(self.__config__.params['turbine_x'], self.__config__.params['turbine_y']) / djl2 / self.scale)
                log(INFO, "Set automatic scaling factor to %e." % self.automatic_scaling_factor)

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
        config = self.__config__
        res = []
        if config.params["turbine_parametrisation"] == "smeared":
            res = numpy.zeros(config.turbine_function_space.dim())

        else:
            if 'turbine_friction' in config.params["controls"] or 'dynamic_turbine_friction' in config.params["controls"]:
                res += numpy.reshape(config.params['turbine_friction'], -1).tolist()

            if 'turbine_pos' in config.params["controls"]:
                res += numpy.reshape(config.params['turbine_pos'], -1).tolist()

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
