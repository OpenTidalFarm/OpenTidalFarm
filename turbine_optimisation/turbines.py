import numpy
import configuration
import math
import numpy
from dolfin import *
from dolfin_adjoint import *
from math import log
from helpers import info, info_green, info_red, info_blue

def turbines_expression(params, derivative_index_selector=-1,  derivative_var_selector=None):
  ''' If the derivative selector is i >= 0, the Expression will compute the derivative of the turbine with index i with respect 
      to either the x or y coorinate or its friction parameter. '''
  code = '''
  class MyFunc : public Expression
  {
  public: 
 
    MyFunc() : Expression()  
    {
    }

    double eval_unit(const double *x2, const double *y2) const
      {
        if ((*x2 < 1.0) && (*y2 < 1.0))
           return exp(-1.0/(1.0-*x2) - 1.0/(1.0-*y2) + 2);
        else
           return 0.0;
      }

    void eval(Array<double>& values, const Array<double>& x,
              const ufc::cell& c) const
      {
        double friction = 0.0;
        const double x2 = x[0]*x[0];
        const double y2 = x[1]*x[1];

        values[0] = eval_unit(&x2, &y2); 
      }

  };
  '''
  return Expression(code)

class Turbines(object):

    def __init__(self, V, params, derivative_index_selector=-1):
        self.params = configuration.Parameters(params)
  
        # Precompute some turbine parameters for efficiency. 
        self.x_pos = numpy.array(params["turbine_pos"])[:,0] 
        self.x_pos_low = self.x_pos-params["turbine_x"]/2
        self.x_pos_high = self.x_pos+params["turbine_x"]/2
  
        self.y_pos = numpy.array(params["turbine_pos"])[:,1] 
        self.y_pos_low = self.y_pos-params["turbine_y"]/2
        self.y_pos_high = self.y_pos+params["turbine_y"]/2
  
        self.x = interpolate(Expression("x[0]"), V).vector().array()
        self.y = interpolate(Expression("x[1]"), V).vector().array()
        self.V = V

    def __call__(self, derivative_index_selector=-1,  derivative_var_selector=None):
        ''' If the derivative selector is i >= 0, the Expression will compute the derivative of the turbine with index i with respect 
          to either the x or y coorinate or its friction parameter. '''
        i = 0
        V = self.V
        # x_unit = (self.x - self.x_pos[i])/ 0.5# / (0.5*self.params["turbine_x"])
        # y_unit = (self.y - self.x_pos[i]) /# / (0.5*self.params["turbine_y"])
        x_unit = self.x 
        y_unit = self.y 
        # We dont mind division by zero
        numpy.seterr(divide = 'ignore')
        ff = numpy.exp(-1/(1-(self.x**2) - 1/(1-self.y**2)+2))
        numpy.seterr(divide = 'warn')

        f = Function(V)
        f.vector().set_local(ff) 

        return f

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
              "turbine_pos": [[0.5, 0.5]],
              "turbine_x": 0.5, 
              "turbine_y": 0.5, 
              "turbine_model": 'BumpTurbine', 
             }

    #f = Function(V)
    #turbines = turbines_expression(params)
    #turbines = turbines_expression(params)
    #turbines = Turbines(params)
    #of.interpolate(turbines)
    turbines = Turbines(V, params)
    f = turbines()
    out_file = File("t.pvd", "compressed")
    out_file << f
