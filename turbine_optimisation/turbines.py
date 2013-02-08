import numpy
import configuration
import math
from dolfin import *
from dolfin_adjoint import *
from math import log
from helpers import info, info_green, info_red, info_blue

class OldTurbines(Expression):
    print_warning = True

    def __init__(self, params, derivative_index_selector=-1,  derivative_var_selector=None, *args, **kwargs):
      ''' If the derivative selector is i >= 0, the Expression will compute the derivative of the turbine with index i with respect 
          to either the x or y coorinate or its friction parameter. '''
      self.params = configuration.Parameters(params)
      self.derivative_index_selector = derivative_index_selector
      self.derivative_var_selector = derivative_var_selector

      # Precompute some turbine parameters for efficiency. 
      self.x_pos = numpy.array(params["turbine_pos"])[:,0] 
      self.x_pos_low = self.x_pos-params["turbine_x"]/2
      self.x_pos_high = self.x_pos+params["turbine_x"]/2

      self.y_pos = numpy.array(params["turbine_pos"])[:,1] 
      self.y_pos_low = self.y_pos-params["turbine_y"]/2
      self.y_pos_high = self.y_pos+params["turbine_y"]/2

      super(Turbines, self).__init__(args, kwargs)

    # The turbine functions will be evaluated between (-1..1) x (-1..1) and should return function values from [0..1].
    def constant_function(self, x):
      '''The turbines are modeled by rectangles with constant friction. '''
      return 1.0

    def constant_derivative(self, x, d):
      if Turbines.print_warning:
        info_red("Warning: The constant turbine is not differentiable at the turbine edges! The derivative will be 0.0")
        Turbines.print_warning = False
      return 0.0

    def gaussian_function(self, x):
      '''The turbines are modeled by a gaussian curve. ''' 
      return exp(-0.5 * (x[0]**2 + x[1]**2) * (-2*math.log(0.05)) )

    def gaussian_derivative(self, x, d):
      ''' This function computes the derivative of the gaussian turbine function with respect to the d'th coordinate. '''
      if Turbines.print_warning:
        info_red("Warning: The gaussian turbine is not differentiable at the turbine edges!")
        Turbines.print_warning = False
      return self.gaussian_function(x) * (-0.5 * (2*x[d]) * (-2*math.log(0.05)) )

    def bump_function(self, x):
      '''The turbines are modeled by the bump function (a smooth function with limited support):
                 /  e**-1/(1-x**2)   for |x| < 1
        psi(x) = |  
                 \  0   otherwise
        For more information see http://en.wikipedia.org/wiki/Bump_function
      '''
      return math.exp(-1.0/(1.0-x[0]**2) - 1.0/(1.0-x[1]**2) + 2)

    def bump_derivative(self, x, d):
      ''' This function computes the derivative of the bump turbine function with respect to the d'th coordinate. '''
      bump = self.bump_function(x)
      bump *= - 2*x[d] / ((1.0-x[d]**2)**2)
      return bump

    def turbine_function(self, params):
      functions = {'GaussianTurbine': self.gaussian_function, 'ConstantTurbine': self.constant_function, 'BumpTurbine': self.bump_function }
      return functions[params['turbine_model']]

    def turbine_derivative(self, params):
      functions = {'BumpTurbine': self.bump_derivative, 'GaussianTurbine': self.gaussian_derivative, 'ConstantTurbine': self.constant_derivative}
      return functions[params['turbine_model']]

    def eval(self, values, x):
        params = self.params
        friction = 0.0
        # Get the cached values
        x_pos = self.x_pos 
        x_pos_low = self.x_pos_low 
        x_pos_high = self.x_pos_high 
        y_pos = self.y_pos
        y_pos_low = self.y_pos_low 
        y_pos_high = self.y_pos_high

        if len(params["turbine_pos"]) > 0:

          # active_turbines is a boolean array that whose i'th element is true if the ith turbine is present at point x
          active_turbines = (x_pos_low < x[0]) & (x_pos_high > x[0]) & (y_pos_low < x[1]) & (y_pos_high > x[1])
          active_turbines_indices = numpy.where(active_turbines == True)[0]

          for i in active_turbines_indices:
            if self.derivative_index_selector < 0:
              # Just compute the evaluation
              f = self.turbine_function(params)

              x_unit = (x[0]-x_pos[i]) / (0.5*params["turbine_x"])
              y_unit = (x[1]-y_pos[i]) / (0.5*params["turbine_y"])
              friction += f([x_unit, y_unit])*params["turbine_friction"][i] 

            elif i == self.derivative_index_selector:
              # Compute the derivative with respect to the specified variable
              i = self.derivative_index_selector
              x_unit = (x[0]-x_pos[i]) / (0.5*params["turbine_x"])
              y_unit = (x[1]-y_pos[i]) / (0.5*params["turbine_y"])

              # Now check with which variable we want to take the derivative with respect to.
              var = self.derivative_var_selector
              if var== 'turbine_friction':
                f = self.turbine_function(params)
                friction += f([x_unit, y_unit])
              elif var in ('turbine_pos_x', 'turbine_pos_y'):
                # The coordinate dimension for which the derivative is to be computed
                d = {'turbine_pos_x': 0, 'turbine_pos_y': 1}[var]
                # The turbine extension in that coordinate dimension
                ext = {'turbine_pos_x': "turbine_x", 'turbine_pos_y': "turbine_y"}[var]
                f = self.turbine_derivative(params)
                friction += f([x_unit, y_unit], d)*params["turbine_friction"][i]*(-1.0/(0.5*params[ext])) # The last multiplier is the derivative of x_unit due to the chain rule
              else: 
                raise ValueError, "Invalid argument for the derivarive variable selector."

        values[0] = friction 

class Turbines(Expression):

    def __init__(self, params, derivative_index_selector=-1,  derivative_var_selector=None, *args, **kwargs):
      ''' If the derivative selector is i >= 0, the Expression will compute the derivative of the turbine with index i with respect 
          to either the x or y coorinate or its friction parameter. '''
      self.params = configuration.Parameters(params)
      self.derivative_index_selector = derivative_index_selector
      self.derivative_var_selector = derivative_var_selector

      # Precompute some turbine parameters for efficiency. 
      self.x_pos = numpy.array(params["turbine_pos"])[:,0] 
      self.x_pos_low = self.x_pos-params["turbine_x"]/2
      self.x_pos_high = self.x_pos+params["turbine_x"]/2

      self.y_pos = numpy.array(params["turbine_pos"])[:,1] 
      self.y_pos_low = self.y_pos-params["turbine_y"]/2
      self.y_pos_high = self.y_pos+params["turbine_y"]/2

      super(Turbines, self).__init__(args, kwargs)

    # The turbine functions will be evaluated between (-1..1) x (-1..1) and should return function values from [0..1].
    def bump_function(self, x):
      '''The turbines are modeled by the bump function (a smooth function with limited support):
                 /  e**-1/(1-x**2)   for |x| < 1
        psi(x) = |  
                 \  0   otherwise
        For more information see http://en.wikipedia.org/wiki/Bump_function
      '''
      return math.exp(-1.0/(1.0-x[0]**2) - 1.0/(1.0-x[1]**2) + 2)

    def bump_derivative(self, x, d):
      ''' This function computes the derivative of the bump turbine function with respect to the d'th coordinate. '''
      bump = self.bump_function(x)
      bump *= - 2*x[d] / ((1.0-x[d]**2)**2)
      return bump

    def eval(self, values, x):
        params = self.params
        friction = 0.0
        # Get the cached values
        x_pos = self.x_pos 
        x_pos_low = self.x_pos_low 
        x_pos_high = self.x_pos_high 
        y_pos = self.y_pos
        y_pos_low = self.y_pos_low 
        y_pos_high = self.y_pos_high

        if len(params["turbine_pos"]) > 0:

          # active_turbines is a boolean array that whose i'th element is true if the ith turbine is present at point x
          active_turbines = (x_pos_low < x[0]) & (x_pos_high > x[0]) & (y_pos_low < x[1]) & (y_pos_high > x[1])
          active_turbines_indices = numpy.where(active_turbines == True)[0]

          for i in active_turbines_indices:
            if self.derivative_index_selector < 0:
              # Just compute the evaluation
              x_unit = (x[0]-x_pos[i]) / (0.5*params["turbine_x"])
              y_unit = (x[1]-y_pos[i]) / (0.5*params["turbine_y"])
              friction += self.bump_function([x_unit, y_unit])*params["turbine_friction"][i] 

            elif i == self.derivative_index_selector:
              # Compute the derivative with respect to the specified variable
              i = self.derivative_index_selector
              x_unit = (x[0]-x_pos[i]) / (0.5*params["turbine_x"])
              y_unit = (x[1]-y_pos[i]) / (0.5*params["turbine_y"])

              # Now check with which variable we want to take the derivative with respect to.
              var = self.derivative_var_selector
              if var == 'turbine_friction':
                friction += self.bump_function([x_unit, y_unit])
              elif var in ('turbine_pos_x', 'turbine_pos_y'):
                # The coordinate dimension for which the derivative is to be computed
                d = {'turbine_pos_x': 0, 'turbine_pos_y': 1}[var]
                # The turbine extension in that coordinate dimension
                ext = {'turbine_pos_x': "turbine_x", 'turbine_pos_y': "turbine_y"}[var]
                friction += self.bump_derivative([x_unit, y_unit], d)*params["turbine_friction"][i]*(-1.0/(0.5*params[ext])) # The last multiplier is the derivative of x_unit due to the chain rule
              else: 
                raise ValueError, "Invalid argument for the derivarive variable selector."

        values[0] = friction 


class TurbineCache:
    def __init__(self):
        self.cache = {}
        self.params = None

    def update(self, config):
        ''' Creates a list of all turbine function/derivative interpolations. This list is used as a cache 
          to avoid the recomputation of the expensive interpolation of the turbine expression. '''
        # If the parameters have not changed, then there is no need to do anything
        if self.params != None:
            if (self.params["turbine_friction"] == config.params["turbine_friction"]).all() and (self.params["turbine_pos"] == config.params["turbine_pos"]).all(): 
                info_green("Skipping turbine cache update")
                return 

        info_green("Updating turbine cache")

        # Store the new turbine paramaters
        self.params = configuration.Parameters(config.params)
        self.params["turbine_friction"] = numpy.copy(config.params["turbine_friction"])
        self.params["turbine_pos"] = numpy.copy(config.params["turbine_pos"])

        # Precompute the interpolation of the friction function of all turbines
        turbines = Turbines(self.params)
        tf = Function(config.turbine_function_space, name = "functional_turbine_friction") 
        tf.interpolate(turbines)
        self.cache["turbine_field"] = tf

        # Precompute the interpolation of the friction function for each individual turbine
        if self.params["print_individual_turbine_power"]:
            info_green("Building individual turbine power friction functions for caching purposes...")
            self.cache["turbine_field_individual"] = [] 
            for i in range(len(self.params["turbine_friction"])):
                params_cpy = configuration.Parameters(self.params)
                params_cpy["turbine_pos"] = [self.params["turbine_pos"][i]]
                params_cpy["turbine_friction"] = [self.params["turbine_friction"][i]]
                turbine = Turbines(params_cpy)
                tf = Function(config.turbine_function_space, name = "functional_turbine_friction") 
                tf.interpolate(turbine)
                self.cache["turbine_field_individual"].append(tf)
                info_green("finished")

        # Precompute the derivatives with respect to the friction magnitude of each turbine
        if "turbine_friction" in self.params["controls"]:
            self.cache["turbine_derivative_friction"] = []
            for n in range(len(self.params["turbine_friction"])):
                turbines = Turbines(self.params, derivative_index_selector = n, derivative_var_selector = 'turbine_friction')
                tfd = Function(config.turbine_function_space, name = "functional_turbine_friction_derivative_with_respect_friction_magnitude_of_turbine_" + str(n)) 
                tfd.interpolate(turbines)
                self.cache["turbine_derivative_friction"].append(tfd)

        # Precompute the derivatives with respect to the turbine position
        if "turbine_pos" in self.params["controls"]:
            self.cache["turbine_derivative_pos"] = []
            for n in range(len(self.params["turbine_pos"])):
                self.cache["turbine_derivative_pos"].append({})
                for var in ('turbine_pos_x', 'turbine_pos_y'):
                    turbines = Turbines(self.params, derivative_index_selector = n, derivative_var_selector = var)
                    tfd = Function(config.turbine_function_space, name = "functional_turbine_friction_derivative_with_respect_position_of_turbine_" + str(n))
                    tfd.interpolate(turbines)
                    self.cache["turbine_derivative_pos"][-1][var] = tfd

class TurbineCache:
    def __init__(self):
        self.cache = {}
        self.params = None

    def update(self, config):
        ''' Creates a list of all turbine function/derivative interpolations. This list is used as a cache 
          to avoid the recomputation of the expensive interpolation of the turbine expression. '''
        # If the parameters have not changed, then there is no need to do anything
        if self.params != None:
            if (self.params["turbine_friction"] == config.params["turbine_friction"]).all() and (self.params["turbine_pos"] == config.params["turbine_pos"]).all(): 
                info_green("Skipping turbine cache update")
                return 

        info_green("Updating turbine cache")

        # Store the new turbine paramaters
        self.params = configuration.Parameters(config.params)
        self.params["turbine_friction"] = numpy.copy(config.params["turbine_friction"])
        self.params["turbine_pos"] = numpy.copy(config.params["turbine_pos"])

        # Precompute the interpolation of the friction function of all turbines
        turbines = Turbines(self.params)
        tf = Function(config.turbine_function_space, name = "functional_turbine_friction") 
        tf.interpolate(turbines)
        self.cache["turbine_field"] = tf

        # Precompute the interpolation of the friction function for each individual turbine
        if self.params["print_individual_turbine_power"]:
            info_green("Building individual turbine power friction functions for caching purposes...")
            self.cache["turbine_field_individual"] = [] 
            for i in range(len(self.params["turbine_friction"])):
                params_cpy = configuration.Parameters(self.params)
                params_cpy["turbine_pos"] = [self.params["turbine_pos"][i]]
                params_cpy["turbine_friction"] = [self.params["turbine_friction"][i]]
                turbine = Turbines(params_cpy)
                tf = Function(config.turbine_function_space, name = "functional_turbine_friction") 
                tf.interpolate(turbine)
                self.cache["turbine_field_individual"].append(tf)
                info_green("finished")

        # Precompute the derivatives with respect to the friction magnitude of each turbine
        if "turbine_friction" in self.params["controls"]:
            self.cache["turbine_derivative_friction"] = []
            for n in range(len(self.params["turbine_friction"])):
                turbines = Turbines(self.params, derivative_index_selector = n, derivative_var_selector = 'turbine_friction')
                tfd = Function(config.turbine_function_space, name = "functional_turbine_friction_derivative_with_respect_friction_magnitude_of_turbine_" + str(n)) 
                tfd.interpolate(turbines)
                self.cache["turbine_derivative_friction"].append(tfd)

        # Precompute the derivatives with respect to the turbine position
        if "turbine_pos" in self.params["controls"]:
            self.cache["turbine_derivative_pos"] = []
            for n in range(len(self.params["turbine_pos"])):
                self.cache["turbine_derivative_pos"].append({})
                for var in ('turbine_pos_x', 'turbine_pos_y'):
                    turbines = Turbines(self.params, derivative_index_selector = n, derivative_var_selector = var)
                    tfd = Function(config.turbine_function_space, name = "functional_turbine_friction_derivative_with_respect_position_of_turbine_" + str(n))
                    tfd.interpolate(turbines)
                    self.cache["turbine_derivative_pos"][-1][var] = tfd

if __name__ == "__main__":
    mesh = UnitSquareMesh(100, 100)
    V = FunctionSpace(mesh, "CG", 1)
    
    params = {"turbine_friction": [0.1, 0.2],
              "turbine_pos": [[0.1, 0.1], [0.5, 0.5]],
              "turbine_x": 0.1, 
              "turbine_y": 0.1, 
              "turbine_model": 'BumpTurbine', 
             }

    f = Function(V)
    turbines = Turbines(params)
    f.interpolate(turbines)
    plot(f, interactive = True)
