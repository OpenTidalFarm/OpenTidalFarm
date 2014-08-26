import numpy
import memoize
import helpers
import sys
import dolfin_adjoint
from dolfin import *
from dolfin_adjoint import *
from turbines import *
from solvers import Solver
import os.path


class ReducedFunctionalNumPy(dolfin_adjoint.ReducedFunctionalNumPy):

    def __init__(self, config, solver, scale=1.0,
                 save_functional_values=False):
        ''' scale is ignored if automatic_scaling is active. '''
        # Hide the configuration since changes would break the memoize algorithm.

        if not isinstance(solver, Solver):
            raise ValueError, "solver must be of type Solver."

        self.__config__ = config
        self.scale = scale
        self.solver = solver
        self.automatic_scaling_factor = None
        self.save_functional_values = save_functional_values
        # Caching variables that store which controls the last forward run was performed
        self.last_m = None
        self.last_state = None
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

            # Get initial conditions
            state = Function(config.function_space, name="Current_state")

            if False and not solver.problem._is_transient and self.last_state is not None:
                # Speed up the nonlinear solves by starting the Newton solve with the most recent state solution
                state.assign(self.last_state, annotate=False)
            else:
                ic = config.params['initial_condition']
                state.assign(ic, annotate=False)

            # Solve the shallow water system
            functional = config.functional(config)
            j = solver.solve(state, functional=functional, turbine_field=tf, annotate=annotate)
            self.last_state = state

            return j

        def compute_gradient(m, forget=True):
            ''' Compute the functional gradient for the turbine positions/frictions array '''

            # If any of the parameters changed, the forward model needs to re-run
            if numpy.any(m != self.last_m):
                compute_functional(m, annotate=True)

            state = self.last_state
            functional = config.functional(config)

            # Output power
            if self.solver.parameters.dump_period > 0:
                if config.params['output_turbine_power']:
                    turbines = self.__config__.turbine_cache.cache["turbine_field"]
                    self.power_file << project(functional.power(state, turbines), config.turbine_function_space, annotate=False)

            # The functional depends on the turbine friction function which we do not have on scope here.
            # But dolfin-adjoint only cares about the name, so we can just create a dummy function with the desired name.
            dummy_tf = Function(FunctionSpace(state.function_space().mesh(), "R", 0), name="turbine_friction")

            if (not solver.problem._is_transient or
                solver.problem.parameters.functional_final_time_only):
                J = Functional(functional.Jt(state, dummy_tf) * dt[FINISH_TIME])

            elif solver.problem.parameters.functional_quadrature_degree == 0:
                # Pseudo-redo the time loop to collect the necessary timestep information
                t = float(solver.problem.parameters.start_time)
                timesteps = [t]
                while (t < float(solver.problem.parameters.finish_time)):
                    t += float(solver.problem.parameters.dt)
                    timesteps.append(t)

                if not solver.problem.parameters.include_time_term:
                    timesteps.pop(0)

                # Construct the functional
                J = Functional(sum(functional.Jt(state, dummy_tf) * dt[t] for t in timesteps))

            else:
                J = Functional(functional.Jt(state, dummy_tf) * dt)

            if 'dynamic_turbine_friction' in config.params["controls"]:
                parameters = [InitialConditionParameter("turbine_friction_cache_t_%i" % i) for i in range(len(config.params["turbine_friction"]))]

            else:
                parameters = InitialConditionParameter("turbine_friction_cache")

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

        def compute_hessian_action(m, m_dot):
            if numpy.any(m != self.last_m):
                self.run_adjoint_model_mem(m, forget=False)

            state = self.last_state

            functional = config.functional(config)
            if (not solver.problem.parameters._transient or
                solver.problem.parameters.functional_final_time_only):
                J = Functional(functional.Jt(state) * dt[FINISH_TIME])
            else:
                J = Functional(functional.Jt(state) * dt)

            H = drivers.hessian(J, InitialConditionParameter("friction"), warn=False)
            m_dot = project(Constant(1), config.turbine_function_space)
            return H(m_dot)

        # For smeared turbine parametrisations we only want to store the 
        # hash of the control values into the pickle datastructure
        hash_keys = (config.params["turbine_parametrisation"] == "smeared")

        self.compute_functional_mem = memoize.MemoizeMutable(compute_functional, hash_keys)
        self.compute_gradient_mem = memoize.MemoizeMutable(compute_gradient, hash_keys)
        self.compute_hessian_action_mem = memoize.MemoizeMutable(compute_hessian_action, hash_keys)

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

        log(INFO, 'Runtime: ' + str(timer.value()) + " s")
        log(INFO, 'j = ' + str(j))
        self.last_j = j

        if self.__config__.params['automatic_scaling']:
            if not self.automatic_scaling_factor:
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

        if self.save_functional_values and MPI.process_number() == 0:
            with open("functional_values.txt", "a") as functional_values:
                functional_values.write(str(self.last_j) + "\n")

        if self.__config__.params["save_checkpoints"]:
            self.save_checkpoint("checkpoint")

        # Compute the scaling factor if never done before
        if self.__config__.params['automatic_scaling'] and not self.automatic_scaling_factor:
            if not 'turbine_pos' in self.__config__.params['controls']:
                raise NotImplementedError("Automatic scaling only works if the turbine positions are control parameters")

            if len(self.__config__.params['controls']) > 1:
                assert(len(dj) % 3 == 0)
                # Exclude the first third from the automatic scaling as it contains the friction coefficients
                djl2 = max(abs(dj[len(dj) / 3:]))
            else:
                djl2 = max(abs(dj))

            if djl2 == 0:
                raise ValueError("Automatic scaling failed: The gradient at the parameter point is zero")
            else:
                self.automatic_scaling_factor = abs(self.__config__.params['automatic_scaling_multiplier'] * max(self.__config__.params['turbine_x'], self.__config__.params['turbine_y']) / djl2 / self.scale)
                log(INFO, "The automatic scaling factor was set to " + str(self.automatic_scaling_factor * self.scale) + ".")

        log(INFO, 'Runtime: ' + str(timer.stop()) + " s")
        log(INFO, '|dj| = ' + str(numpy.linalg.norm(dj)))

        if self.__config__.params['automatic_scaling']:
            return dj * self.scale * self.automatic_scaling_factor
        else:
            return dj * self.scale

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

    def hessian(self, m_array, m_dot_array):
        ''' Interface function for dolfin_adjoint.ReducedFunctional '''

        raise NotImplementedError('The Hessian computation is not yet implemented')

    def obj_to_array(self, obj):

        return dolfin_adjoint.optimization.get_global(obj)

    def set_parameters(self, m_array):

        m = [p.data() for p in self.parameter]
        dolfin_adjoint.optimization.set_local(m, m_array)


class ReducedFunctional(ReducedFunctionalNumPy):
    pass
