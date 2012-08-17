import numpy
import memoize
import shallow_water_model as sw_model
import helpers 
import sys
from animated_plot import AnimatedPlot
from functionals import DefaultFunctional
from dolfin import *
from turbines import *
from numpy.linalg import norm
from helpers import info, info_green, info_red, info_blue

class ReducedFunctional:
    def __init__(self, config, scaling_factor = 1.0, forward_model = sw_model.sw_solve, plot = False):
        ''' If plot is True, the functional values will be automatically saved in a plot.
            scaling_factor is ignored if automatic_scaling is active. '''
        # Hide the configuration since changes would break the memoize algorithm. 
        self.__config__ = config
        self.scaling_factor = scaling_factor
        self.automatic_scaling_factor = None
        self.plot = plot
        # Caching variables that store for which controls the last forward run was performed
        self.last_m = None
        self.last_djdm = None
        self.last_state = None
        self.turbine_file = File("turbines.pvd", "compressed")

        if plot:
           self.plotter = AnimatedPlot(xlabel = "Iteration", ylabel = "Functional value")

        def run_forward_model(m, return_final_state = False):
            ''' This function solves the forward and adjoint problem and returns the functional value and its gradient for the parameter choice m. ''' 
            adj_reset()

            # Change the control variables to the config parameters
            shift = 0
            if 'turbine_friction' in config.params['controls']: 
                shift = len(config.params["turbine_friction"])
                config.params["turbine_friction"] = m[:shift]
            if 'turbine_pos' in config.params['controls']: 
                mp = m[shift:]
                config.params["turbine_pos"] = numpy.reshape(mp, (-1, 2))

            parameters["adjoint"]["record_all"] = True 
            self.last_m = m

            # Get initial conditions
            state = Function(config.function_space, name="Current_state")
            state.interpolate(config.params['initial_condition'](config)())

            # Set up the turbine field 
            tf = Function(self.__config__.turbine_function_space, name = "friction") 
            tf.interpolate(Turbines(config.params))
            info_green("Turbine integral: %f ", assemble(tf*dx))
            info_green("The correct integral should be: %f ",  25.2771) # computed with wolfram alpha using:
            # int 0.17353373* (exp(-1.0/(1-(x/10)**2)) * exp(-1.0/(1-(y/10)**2)) * exp(2)) dx dy, x=-10..10, y=-10..10 
            info_red("relative error: %f", (assemble(tf*dx)-25.2771)/25.2771)
            self.turbine_file << tf

            # Solve the shallow water system
            functional = DefaultFunctional(config.function_space, config)
            j, self.last_djdm = forward_model(config, state, functional=functional, turbine_field = tf)
            self.last_state = state

            if return_final_state:
                return j, state
            else:
                return j 

        def run_adjoint_model(m):
            # If the last forward run was performed with the same parameters, then all recorded values by dolfin-adjoint are still valid for this adjoint run
            # and we do not have to rerun the forward model.
            if numpy.any(m != self.last_m):
                run_forward_model(m)

            state = self.last_state
            djdm = self.last_djdm

            functional = DefaultFunctional(config.function_space, config)
            if config.params['steady_state'] or config.params["functional_final_time_only"]:
                J = Functional(functional.Jt(state)*dt[FINISH_TIME])
            else:
                J = Functional(functional.Jt(state)*dt)
            djdudm = compute_gradient(J, InitialConditionParameter("friction"))

            # Let J be the functional, m the parameter and u the solution of the PDE equation F(u) = 0.
            # Then we have 
            # dJ/dm = (\partial J)/(\partial u) * (d u) / d m + \partial J / \partial m
            #               = adj_state * \partial F / \partial u + \partial J / \partial m
            # In this particular case m = turbine_friction, J = \sum_t(ft) 
            dj = [] 
            U = config.function_space.split()[0].sub(0) # Extract the first component of the velocity function space 
            U = U.collapse() # Recompute the DOF map
            tfd = Function(self.__config__.turbine_function_space, name = "friction_derivative") 
            if 'turbine_friction' in config.params["controls"]:
                # Compute the derivatives with respect to the turbine friction
                for n in range(len(config.params["turbine_friction"])):
                    tfd.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector='turbine_friction'))
                    dj.append( djdudm.vector().inner(tfd.vector()) )

            if 'turbine_pos' in config.params["controls"]:
                # Compute the derivatives with respect to the turbine position
                for n in range(len(config.params["turbine_pos"])):
                    for var in ('turbine_pos_x', 'turbine_pos_y'):
                        tfd.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector=var))
                        dj.append( djdudm.vector().inner(tfd.vector()) )
            dj = numpy.array(dj)  
            
            # Now add the \partial J / \partial m term
            dj += djdm

            return dj 

        self.run_forward_model_mem = memoize.MemoizeMutable(run_forward_model)
        self.run_adjoint_model_mem = memoize.MemoizeMutable(run_adjoint_model)
        
    def j(self, m):
        ''' This memoised function returns the functional value for the parameter choice m. '''
        timer = dolfin.Timer("dj Evaluation") 
        timer.start()
        j = self.run_forward_model_mem(m) 
        timer.stop()
        if self.plot:
            self.plotter.addPoint(j)
            self.plotter.savefig("functional_plot.png")
        info_green('Evaluating j(' + m.__repr__() + ') = ' + str(j))
        info_blue('Runtime: ' + str(timer.value())  + " s")

        if self.__config__.params['automatic_scaling']:
            if not self.automatic_scaling_factor:
                # Computing dj will set the automatic scaling factor. 
                info_blue("Computing derivative to determine the automatic scaling factor")
                dj = self.dj(m)
            info_green('Scaled j(' + m.__repr__() + ') = ' + str(self.automatic_scaling_factor * self.scaling_factor * j))
            return j * self.scaling_factor * self.automatic_scaling_factor
        else:
            info_green('Scaled j(' + m.__repr__() + ') = ' + str(self.scaling_factor * j))
            return j * self.scaling_factor

    def dj(self, m):
        ''' This memoised function returns the gradient of the functional for the parameter choice m. '''
        timer = dolfin.Timer("dj Evaluation") 
        timer.start()
        dj = self.run_adjoint_model_mem(m)
        timer.stop()

        # Compute the scaling factor if never done before
        if self.__config__.params['automatic_scaling'] and not self.automatic_scaling_factor:
            if not self.__config__.params['controls'] == ['turbine_pos']:
                raise NotImplementedError, "Automatic scaling works currently only if the turbine positions are the only parameters"

            djl2 = max(abs(dj))
            if djl2 == 0:
                raise ValueError, "Automatic scaling failed: The gradient at the parameter point is zero"
            else:
                self.automatic_scaling_factor = abs(self.__config__.params['automatic_scaling_multiplier'] * max(self.__config__.params['turbine_x'], self.__config__.params['turbine_y']) / djl2 / self.scaling_factor)
                info_blue("The automatic scaling factor was set to " + str(self.automatic_scaling_factor * self.scaling_factor) + ".")

        info_green('Evaluating dj(' + m.__repr__() + ') = ' + str(dj)) 
        info_blue('Runtime: ' + str(timer.value())  + " s")
        if self.__config__.params['automatic_scaling']:
            info_green('Scaled dj(' + m.__repr__() + ') = ' + str(self.automatic_scaling_factor * self.scaling_factor * dj))
            return dj * self.scaling_factor * self.automatic_scaling_factor
        else:
            info_green('Scaled dj(' + m.__repr__() + ') = ' + str(self.scaling_factor * dj))
            return dj * self.scaling_factor

    def dj_with_check(self, m, seed = 0.1, tol = 1.8):
        ''' This function checks the correctness and returns the gradient of the functional for the parameter choice m. '''

        info_red("Checking derivative at m = " + str(m))
        p = numpy.random.rand(len(m))
        minconv = helpers.test_gradient_array(self.j, self.dj, m, seed = seed, perturbation_direction = p)
        if minconv < tol:
            info_red("The gradient taylor remainder test failed.")
            sys.exit(1)
        else:
            info_green("The gradient taylor remainder test passed.")

        return self.dj(m)

    def initial_control(self):
        ''' This function returns the control variable array that derives from the initial configuration. '''
        config = self.__config__ 
        res = []
        if 'turbine_friction' in config.params["controls"]:
            res += config.params['turbine_friction'].tolist()
        if 'turbine_pos' in config.params["controls"]:
            res += numpy.reshape(config.params['turbine_pos'], -1).tolist()
        return numpy.array(res)

