"""
.. module:: ShallowWaterReducedFunctional
    :synopsis: Defines the ShallowWaterReducedFunctional class which interfaces
    the shallow water solver and shallow water objectives with the
    MasterReducedFunctional class.

"""

import numpy
from generic_reduced_functional import GenericReducedFunctional

        #!!!TODO!!!
        #parameters for this class to include:
        #automatic scaling option (on/off) formerly params['automatic_scaling']
        #manual scaling (value) formerly keyword arg 'scale'
        #functional quadrature degree (int) formerly config.params['functional_quadrature_degree']
        #solver params like start_time, finish_time, dt, include_time_term
        # smeared parametrisation?
        #turbine field cache self.farm._turbine_cache.field
        #parametrisation
        #output options

class ReducedFunctional(object):

    def reduced_functional(self, m):
        """returns the reduced functional from the turbine locations

        :param m: cartesian co-ordinates of the turbines.
        :type m: TODO.

        .. note::

            This is method is the top of many layers; m is passed in and it is
            converted to a friction function from which the functional is
            calculated - these steps are memoized. The functional is then scaled
            as required before being returned here.
        """
        return self._scaled_reduced_functional(m)

    def _scaled_reduced_functional(self, m):
        """this determines the reduced functional via !!!TODO!!! and,
        if automatic or manual scaling is activated, this method scales the
        reduced functional. It also saves a checkpoint as required.

        :param m: cartesian co-ordinates of the turbines.
        :type m: TODO.
        """
        # Start the timer
        info_green('Start evaluation of the functional')
        timer = dolfin.Timer('j evaluation')

        #Call other methods to actually calculate the reduced functional
        reduced_functional = self._memoized_reduced_functional(m,
                annotate=annotate)

        # Stop the timer and print some diagnostics
        timer.stop()
        info_blue('Runtime: ' + str(timer.value()) + " s")
        info_green('j = ' + str(j))

        #Save a checkpoint if required
        if self.checkpoint:
            self._save_checkpoint("checkpoint")

        # Deal with any scaling we need
        self.scaling_factor = self._manage_scaling_factors()
        return reduced_functional * self._manual_scaling_factor *
                    self._automatic_scaling_factor
        else: #if we're not doing any automatic scaling
            return reduced_functional * self._manual_scaling_factor

    def _reduced_functional_from_m(self, m, annotate=True):
        """converts m into a turbine friction function and returns the reduced
        funtional via _reduced_functional_from_tf method

        :param m: cartesian co-ordinates of the turbines.
        :type m: TODO.
        """
        # Cache the last m
        self.last_m = m
        self._update_turbine_cache(m)

        # Convert m to a turbine friction function and pass it along
        # !!!TODO!!! check this (below)
        tf = self._turbine_cache.field
        m = self.farm.deserialize(self._controls)
        return self._reduced_function_from_tf(tf, annotate=annotate)

    def _reduced_functional_from_tf(self, tf, annotate=True):
        """calls the shallow water solver to calculate u, then ships this off
        along with the turbine friction function to the for the objective
        functional to be computed

        :param tf: turbine friction function.
        :type tf: TODO.
        """
        # Reset the dolfin_adjoint tape
        adj_reset()
        dolfin.parameters['adjoint']['record_all'] = True
        # !!!TODO!!! if config.params['revolve_parameters'] is not None ??????

        # Initialise the state for our solve - either with initial conditions,
        # or with the results of our last solve.
        state = self._initialise_state()

        # Solve the shallow water equations
        functional = self._sw_solver(problem, state, objective, tf,
                annotate = annotate)

        # Cache the state and return the functional
        self.last_state = state
        return functional


class Derivative(object):

    def derivative(self, m_array, taylor_test=False, seed=0.001, forget=True,
            **kwargs):
        """returns the functional derivative from the turbine locations and
        can perform a Taylor test

        :param m_array: cartesian co-ordinates of the turbines.
        :type m_array: TODO.
        :param taylor_test: Performs a Taylor test to check the gradient
        :type taylor_test: Boolean
        :param seed: seed the Taylor test
        :type seed: float
        :param forget: !!!TODO!!!
        :type forget: Boolean

        .. note::

            Like the reduced functional, this method is the top of many layers.
        """
        if taylor_test:
            return self._functional_derivative_with_check(m_array, seed, forget)
        else:
            return self._scaled_derivative(m_array, forget)

    def _scaled_derivative(self, m_array, forget, optimisation_iteration=True):
        """this determines the functional derivative via !!!TODO!!! and,
        if automatic or manual scaling is activated, this method scales the
        reduced functional. It also saves a checkpoint as required.

        :param m_array: cartesian co-ordinates of the turbines.
        :type m_array: TODO.
        :param forget: !!!TODO!!!
        :type forget: Boolean
        :param optimisation_iteration: !!!TODO!!!
        :type optimisation_iteration: Boolean
        """
        # Start the timer
        info_green('Start evaluation of functional derivative')
        timer = dolfin.Timer('dj evaluation')

        functional_derivative = self._memoized_derivative(m_array, forget)

        # Stop the timer and print some diagnostics
        info_blue('Runtime: ' + str(timer.stop()) + " s")
        info_green('|dj| = ' + str(numpy.linalg.norm(dj)))

        # This is an opportune moment to increment our iteration counter and do
        # some outputting and housekeeping.
        self._housekeeping(m)

        #Save a checkpoint if required
        if self.checkpoint:
            self._save_checkpoint("checkpoint")

        # Check the scaling is all set up and ready
        self._manage_scaling_factors()

        return functional_derivative * self.scaling_factor

    def _derivative_from_m(self, m_array, forget=True):
         """this determines the functional derivative from m

        :param m_array: cartesian co-ordinates of the turbines.
        :type m_array: TODO.
        :param forget: !!!TODO!!!
        :type forget: Boolean
        """
        # Check if parameters have changed - and if so rerun the forward model
        if numpy.any(m_array != self.last_m):
            self._compute_functional_from_m(m_array)

        # Get the last state
        state = self.last_state

        # Output the power plots to vtu
        self._output_power()

        # The functional depends on the turbine friction function which we do
        # not have on scope here. But dolfin-adjoint only cares about the name,
        # so we can just create a dummy function with the desired name.
        dummy_tf = Function(FunctionSpace(state.function_space().mesh(), "R",
            0), name="turbine_friction")

        #Determine what comprises our functional e.g. is it a sum over time?
        if steady_state or functional_final_time_only:
            functional = Functional(self.objective.Jt(state, dummy_tf) *
                    dt[FINISH_TIME])
        elif functional_quadrature_degree == 0:
            # We need to collect the necessary timestep info by pseudo redoing
            # the time loop.
            t = float(self._sw_problem.start_time)
            timesteps = [t]
            while (t < float(self._sw_problem.finish_time)):
                t += float(self._sw_problem.dt)
                timesteps.append(t)
            functional = Functional(sum(self.objective.Jt(state, dummy_tf) * dt[t]
                for t in timesteps))
        else:
            if not include_time_term:
                raise NotImplementedError, "Multi-steady state simulations only
                    work with 'functional_quadrature_degree=0' or
                    'functional_final_time_only=True'"
            functional = Functional(self.objective.Jt(state, dummy_tf) * dt

        # !!!TODO!!! No idea what this does...
        if 'dynamic_turbine_friction' in config.params["controls"]:
            parameters = [InitialConditionParameter("turbine_friction_cache_t_%i"
                % i) for i in % range(len(config.params["turbine_friction"]))]
        else:
            parameters = InitialConditionParameter("turbine_friction_cache")

        # Ask dolfin_adjoint for our gradient with respect to our turbine
        # friction field
        dfunctional_dtf = dolfin_adjoint.compute_gradient(functional, parameters,
            forget=forget)
        dolfin.parameters['adjoint']['stop_annotating'] = False

        # Our derivative is dependant on whether we're optimising discrete
        # turbines or a 'smeared' friction field.
        if smeared_parameterisation:
            return dolfin_adjoint.optimization.get_global(dfunctional_dtf)
        else:
            return self._derivative_from_dfunctional_dtf(dfunctional_dtf, TODO)

    def _derivative_from_dfunctional_dtf(self, dfunctional_dtf, TODO):
        """ Returns the derivative of the functional with respect to our control
        parameter as encoded in m - hence this will be an m-long vector rather
        than a field.

        :param dfunctional_dtf: the derivative of the functional with respect to
        the the turbine friction field
        :type dfunctional_dtf: TODO... dolfin.Function...?
        """
        # For convenience let's shorten dfunctional_dtf
        djdtf = dfunctional_dtf
        dfunctional_dm = []

        #!!!TODO!!! lot of config.turbine_cache here...
        # See which controls we have and compute the derivatives
        if self._control.friction:
            # Compute the derivatives with respect to the turbine friction
            for tfd in config.turbine_cache.cache["turbine_derivative_friction"]:
                config.turbine_cache.update(config)
                dfunctional_dm.append(djdtf.vector().inner(tfd.vector()))
        elif self._control.dynamic_friction:
            # Compute the derivatives with respect to the turbine friction
            for djdtf_arr, t in zip(djdtf,
                config.turbine_cache.cache["turbine_derivative_friction"]):
                for tfd in t:
                    config.turbine_cache.update(config)
                    dfunctional_dm.append(djdtf_arr.vector().inner(tfd.vector()))

        if self._control.position:
            # Compute the derivatives with respect to the turbine position
            for d in config.turbine_cache.cache["turbine_derivative_pos"]:
                for var in ('turbine_pos_x', 'turbine_pos_y'):
                    config.turbine_cache.update(config)
                    tfd = d[var]
                    dfunctional_dm.append(djdtf.vector().inner(tfd.vector()))

        return numpy.array(dfunctional_dm)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #   C O M P U T E   H E S S I A N   A C T I O N   # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


    def hessian(self, m, m_dot):
        """ Returns the hessian of the functional with respect to the control
        variable.

        :param m:
        :type m:
        :param m_dot:
        :type m_dot:

        .. note::
            not yet implemented.
        """
        raise NotImplementedError('The Hessian computation is not yet
                implemented')

    def _compute_hessian_action(self, m, m_dot):
        """ Placeholding for future implementation of the hessian.
        """
        return None

class Helpers(object):

    def _manage_scaling_factors(self):
        """check if automatic scaling, or manual scaling of the reduced
        functional (and derivative) is required and if so, combine these into a
        single manageable scaling factor
        """
        #Check if we need to scale
        if self._automatic_scaling: #NOTE can this all be moved to __init__?
            #The automatic scaling factor remains the same for the whole
            # optimisation - so on the first run we need to calculate it. A
            # gradient calculation is required to do this.
            if not self.automatic_scaling_factor:
                info_blue('Computing derivative to determine the automatic
                           scaling factor')
                self._compute_automatic_scaling_factor(m)

    def _compute_automatic_scaling_factor(self, m):
        """calls methods to find the derivative and uses this to determine the
        automatic scaling factor

        :param m: cartesian co-ordinates of the turbines.
        :type m: TODO.
        """
        #!!!TODO!!! replaces the call self.dj(m, forget=False,
        # optimisation_iteration=False) on line 270

    def _initialise_state(self):
        # !!!TODO!!! where does the function space reside?
        # Get initial conditions
        state = Function(self.function_space, name='current_state')

        if config.params["steady_state"] and config.params["include_time_term"]
            and self.last_state is not None:
        # Speed up the nonlinear solves by starting the Newton solve with the
        # most recent state solution
            state.assign(self.last_state, annotate=False)
        else:
            ic = config.params['initial_condition']
            state.assign(ic, annotate=False)

        return state

    def _housekeeping(self, m):
        """ A method to cache the turbines and output some vtu-s.
        """
        if optimisation_iteration:
            self.__config__.optimisation_iteration += 1
            if self.__config__.params["dump_period"] > 0:
            # A cache hit skips the turbine cache update, so we need to trigger
            # it manually...
            if self._memoized_derivative.has_cache(m. forget):
                self.update_turbine_cache(m) #!!!TODO!!! self.farm.update...?
            # Output the vtu-s
            if self._control.dynamic_friction:
                info_red("Turbine VTU output not yet implemented for dynamic
                        turbine control")
            else:
                self.turbine_file <<
                self.__config__.turbine_cache.cache["turbine_field"]
                # Compute the total amount of friction due to the turbines if
                # using the 'smeared' approach.
                if self.smeared_parametrisation:
                    friction =
                    assemble(self.__config__.turbine_cache.cache["turbine_field"]
                            * dx)
                    info_green("Total amount of friction: %f " % friction)

        # Save the functional values to a text file as required
        if self.save_functional_values and MPI.process_number == 0:
            with open("functional_values.txt", "a") as functional_values:
                functional_values.write(str(self.last_j) + "\n")

    def _output_power(self, state):
        """ Output the 'power' field to a vtu !!!TODO!!! is this in fact the
        variation of the combined objective over the domain???
        """
        if config.params["output_turbine_power"]:
            if self._parametrisation.thrust or
                self._parametrisation.implicit_thrust:
                info_red("Turbine power VTU's is not yet implemented with
                        thrust based turbines parameterisations and dynamic
                        turbine friction control.")
            else:
                turbines = self.__config__.turbine_cache.cache["turbine_field"]
                self.power_file << project(self.objective.power(state, turbines),
                        config.turbine_function_space, annotate=False)

    def _save_checkpoint(self, base_filename):
        """ Checkpoint the reduced functional. Checkpoints can be used to
        restart the turbine optimisation.
        """
        base_path = os.path.join(self.__config__.params["base_path"],
                base_filename)
        self._memoized_reduced_functional.save_checkpoint(base_path + "_fwd.dat")
        self._memoized_derivative.save_checkpoint(base_path + "_adj.dat")

    def _load_checkpoint(self, base_filename='checkpoint'):
        ''' Checkpoint the reduceduced functional from which can be used to
        restart the turbine optimisation.
        '''
        base_path = os.path.join(self.__config__.params["base_path"],
                base_filename)
        self.compute_functional_mem.load_checkpoint(base_path + "_fwd.dat")
        self.compute_gradient_mem.load_checkpoint(base_path + "_adj.dat")

    def _update_turbine_cache(self, m):
        """ Reconstructs the control parameters and updates the farm cache
        """
        #!!!TODO!!!


class AdditionalProperties(object):

    def _set_automatic_scaling(self, automatic_scaling):
        self._automatic_scaling = automatic_scaling

    def _get_automatic_scaling(self, automatic_scaling):
        return self._automatic_scaling

    automatic_scaling = property(_get_automatic_scaling, _set_automatic_scaling,
            "automatic scaling - on / off (boolean)"

    def _set_output_turbine_power(self, output_turbine_power):
        self._output_turbine_power = output_turbine_power

    def _get_output_turbine_power(self, output_turbine_power):
        return self._output_turbine_power

    output_turbine_power = property(_get_output_turbine_power,
        _set_output_turbine_power, "output turbine power to vtu - on / off (boolean)"

    def _set_checkpoint(self, checkpoint):
        self._checkpoint = checkpoint

    def _get_checkpoint(self, checkpoint):
        return self._checkpoint

    checkpoint = property(_get_checkpoint, _set_checkpoint, "save checkpoints
        during optimisation - on / off (boolean)"


class ShallowWaterReducedFunctional(GenericReducedFunctional,
                                    ReducedFunctional, Derivative, Hessian,
                                    Helpers, AdditionalProperties):
    """Constructs the reduced functional object for the shallow water solver

    Overloads the GenericReducedFunctional

    .. note::

        The methods which must be overloaded are __init__, reduced_functional
        derivative and hessian.

    """


    def __init__(self, farm, objective):
        """Inits ShallowWaterReducedFunctional class with:

        :param farm: TODO
        :type farm: TODO
        :param functional: The objective functional of the shallow water solver
        :type functional: GenericFunctional class object
        :param control: The control parameter (turbine pos / friction)
        :type

        """
        # Make some things a little more accessible for convenience
        self.farm = farm
        self.functional = functional
        self._controls = objective._controls
        self._sw_problem = objective._solver._problem
        self._sw_solver = objective._solver
        self._parametrisation = farm._prototype_turbine._parametrisation
        self._turbine_cache = farm._turbine_cache
        self.function_space = self._sw_problem._function_space
        m = self.farm.deserialize(self._controls)

        # Set up the memoization wrappers
        self._memoized_reduced_functional = \
            memoize.MemoizeMutable(_reduced_functional_from_m, hash_keys)
        self._memoized_derivative = \
            memoize.MemoizeMutable(_derivative_from_m, hash_keys)
        self._memoized_hessian = \
            memoize.MemoizeMutable(_compute_hessian_action, hash_keys)

        # Set the defaults of some of the additional class parameters
        self._automatic_scaling = 1
        self._output_turbine_power = True
        self._checkpoint = False

        # Initialise some caching variables
        self.last_m = None
        self.last_state = None

        # Do some preparatory work
        if self._output_turbine_power:
            self.power_file = File(config.params["base_path"] + os.path.sep +
                    "power.pvd", "compressed")







































