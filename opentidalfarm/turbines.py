import numpy
from parameter_dict import ParameterDictionary
import math
import numpy
from dolfin import *
from dolfin_adjoint import *
from math import log
from helpers import info, info_green, info_red, info_blue

class Turbines(object):

    def __init__(self, V, params, derivative_index_selector=-1):
        self.params = ParameterDictionary(params)
  
        # Precompute some turbine parameters for efficiency. 
        self.x = interpolate(Expression("x[0]"), V).vector().array()
        self.y = interpolate(Expression("x[1]"), V).vector().array()
        self.V = V

    def __call__(self, name = "", derivative_index_selector=None, derivative_var_selector=None):
        ''' If the derivative selector is i >= 0, the Expression will compute the derivative of the turbine with index i with respect 
          to either the x or y coorinate or its friction parameter. '''
        V = self.V
        params = self.params
  
        if derivative_index_selector == None: 
          turbine_pos = params["turbine_pos"]
          turbine_friction = params["turbine_friction"]
        else:
          turbine_pos = [params["turbine_pos"][derivative_index_selector]]
          turbine_friction = [params["turbine_friction"][derivative_index_selector]]

        ff = numpy.zeros(len(self.x))
        # We dont mind division by zero
        numpy.seterr(divide = 'ignore')
        eps = 1e-12
        for (x_pos, y_pos), friction in zip(turbine_pos, turbine_friction):
          x_unit = numpy.minimum(numpy.maximum((self.x - x_pos) / (0.5*self.params["turbine_x"]), -1+eps), 1-eps) 
          y_unit = numpy.minimum(numpy.maximum((self.y - y_pos) / (0.5*self.params["turbine_y"]), -1+eps), 1-eps) 

          # Apply chain rule to get the derivative with respect to the turbine friction 
          e = numpy.exp(-1/(1-x_unit**2) - 1./(1-y_unit**2)+2)
          if derivative_index_selector == None:
            ff += e * friction

          elif derivative_var_selector == 'turbine_friction':
            ff += e 

          if derivative_var_selector == 'turbine_pos_x':
            ff += e * (-2*x_unit / ((1.0-x_unit**2)**2)) * friction*(-1.0/(0.5*params["turbine_x"])) 

          elif derivative_var_selector == 'turbine_pos_y':
            ff += e * (-2*y_unit / ((1.0-y_unit**2)**2)) * friction*(-1.0/(0.5*params["turbine_y"])) 

        numpy.seterr(divide = 'warn')

        f = Function(V, name = name, annotate = False)
        f.vector().set_local(ff) 
        f.vector().apply("insert")
        return f

class TurbineDomain(SubDomain):
    def __init__(self, params, turbine_index):
        self.center = params["turbine_pos"][turbine_index]
        self.turbine_x = 3*params["turbine_x"]
        self.turbine_y = 3*params["turbine_y"]
        super(TurbineDomain, self).__init__()

    def inside(self, x, on_boundary):
        return (between(x[0]-self.center[0], (-self.turbine_x, self.turbine_x)) 
                and between(x[1]-self.center[1], (-self.turbine_y, self.turbine_y)))

class TurbineCache:
    def __init__(self):
        self.cache = {}
        self.params = None
        self.dx = None 

    def turbine_integral(self):
        ''' Computes the integral of the turbine '''
        unit_bump_int = 1.45661 # Computed with wolfram alpha: integrate e^(-1/(1-x**2)-1/(1-y**2)+2) dx dy, x=-0.999..0.999, y=-0.999..0.999
        return unit_bump_int*self.params["turbine_x"]*self.params["turbine_y"]/4

    def update_measures(self, config):
        ''' Update the integration measures dx[i] for each turbine i '''

        t = Timer("Creating turbine measures")
        self.dx = []
        for i in range(len(config.params["turbine_pos"])):
            # Turbines may overlap, so we need to create a new CellFunction for each turbine.
            domains = CellFunction("size_t", config.domain.mesh)
            domains.set_all(0)

            (TurbineDomain(config.params, i)).mark(domains, 1)
            self.dx.append(Measure("dx")[domains])
        print "Finished creating turbine measures", t.stop()

    def update(self, config):
        ''' Creates a list of all turbine function/derivative interpolations. This list is used as a cache 
          to avoid the recomputation of the expensive interpolation of the turbine expression. '''

        # Update the turbine intergration measures
        self.update_measures(config)

        # If the parameters have not changed, then there is no need to do anything
        if self.params != None:
            if (self.params["turbine_friction"] == config.params["turbine_friction"]).all() and (self.params["turbine_pos"] == config.params["turbine_pos"]).all(): 
                return 

        info_green("Updating turbine cache")

        # Store the new turbine paramaters
        self.params = ParameterDictionary(config.params)
        self.params["turbine_friction"] = numpy.copy(config.params["turbine_friction"])
        self.params["turbine_pos"] = numpy.copy(config.params["turbine_pos"])

        # Precompute the interpolation of the friction function of all turbines
        turbines = Turbines(config.turbine_function_space, self.params)
        tf = turbines(name = "turbine_friction")
        self.cache["turbine_field"] = tf

        # Precompute the derivatives with respect to the friction magnitude of each turbine
        if "turbine_friction" in self.params["controls"]:
            self.cache["turbine_derivative_friction"] = []
            for n in range(len(self.params["turbine_friction"])):
                tfd = turbines(derivative_index_selector = n, 
                               derivative_var_selector = 'turbine_friction', 
                               name = "turbine_friction_derivative_with_respect_friction_magnitude_of_turbine_" + str(n)) 
                self.cache["turbine_derivative_friction"].append(tfd)

        # Precompute the derivatives with respect to the turbine position
        if "turbine_pos" in self.params["controls"]:
            self.cache["turbine_derivative_pos"] = []
            for n in range(len(self.params["turbine_pos"])):
                self.cache["turbine_derivative_pos"].append({})
                for var in ('turbine_pos_x', 'turbine_pos_y'):
                    tfd = turbines(derivative_index_selector = n, 
                                        derivative_var_selector = var,
                                        name = "turbine_friction_derivative_with_respect_position_of_turbine_" + str(n))
                    self.cache["turbine_derivative_pos"][-1][var] = tfd

if __name__ == "__main__":
    mesh = RectangleMesh(-1, -1, 1, 1, 100, 100)
    V = FunctionSpace(mesh, "CG", 1)
    
    params = {"turbine_friction": [0.1, 0.2],
              "turbine_pos": [[0.5, 0.5], [-0.5, -0.5]],
              "turbine_x": 0.5, 
              "turbine_y": 0.5, 
             }

    turbines = Turbines(V, params)
    f = turbines()#1, "turbine_pos_y")
    plot(f, interactive = True)
