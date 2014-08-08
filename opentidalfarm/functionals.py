from dolfin import dot, Constant, dx
from parameter_dict import ParameterDictionary
import shallow_water_model

class GenericFunctional(object):
    '''Generic functional object which should be overloaded by implemented
    functionals
    '''
 
    def __init__(self):
        raise NotImplementedError('GenericFunctional.__init__ needs to be \
                overloaded')
                
    def __add__(self, other):
        ''' method to add functionals together '''
        return CombinedFunctional([self, other])
        
    def __sub__(self, other):
        ''' method to subtract one functional from another '''
        return CombinedFunctional([self, -other])
        
    def __mul__(self, other):
        ''' method to scale a functional '''
        return ScaledFunctional(self, other)
        
    def __rmul__(self, other):
        ''' preserves commutativity of scaling '''
        return ScaledFunctional(self, other)
    
    def __neg__(self):
        ''' implements the negative of the functional '''
        return -1 * self

    def Jt(self, state, tf):
        '''This method should return the form which computes the functional's
        contribution for one timelevel.'''
        raise NotImplementedError('GenericFunctional.Jt needs to be \
                overloaded.')
                
    
class CombinedFunctional(GenericFunctional):
    ''' Constructs a single combined functional by adding one functional to 
    another.
    '''
    
    def __init__(self, functional_list):
        for functionals in functional_list:
            assert isinstance(functionals, GenericFunctional) 
        self.functional_list = functional_list
        
    def Jt(self, state, tf):
        '''Returns the form which computes the combined functional.'''
        combined_functional = sum([functional.Jt(state, tf) for functional in \
            self.functional_list])
        return combined_functional


class ScaledFunctional(GenericFunctional):
    '''Scales the functional 
    '''
    
    def __init__(self, functional, scaling_factor):
        assert isinstance(functional, GenericFunctional) 
        assert isinstance(scaling_factor, int) or isinstance(scaling_factor, float)
        self.functional = functional
        self.scaling_factor = scaling_factor
        
    def Jt(self, state, tf):
        '''Returns the form which computes the combined functional.'''
        scaled_functional = self.scaling_factor * self.functional.Jt(state, tf)
        return scaled_functional


###########################################
### Actually implement some functionals ###
###########################################


class PowerFunctional(GenericFunctional):
    ''' Implements a simple functional of the form:
    J(u, m) = rho * turbines(m) * (||u||**3)
    where turbines(m) defines the friction function due to the turbines.
    '''
    
    def __init__(self, config):
        ''' Constructs a new functional for computing the power. The turbine 
        settings are derived from the settings params. '''
        config.turbine_cache.update(config)
        self.config = config
        # Create a copy of the parameters so that future changes will not 
        # affect the definition of this object.
        self.params = ParameterDictionary(dict(config.params))


    def Jt(self, state, tf):
        return self.power(state, tf) * self.config.site_dx(1)

    def power(self, state, turbines):
        ''' Computes the power field over the domain '''
        return self.params['rho'] * turbines * (dot(state[0], state[0]) + \
                    dot(state[1], state[1])) ** 1.5

    def Jt_individual(self, state, i):
        ''' Computes the power output of the i'th turbine. '''
        tf = self.config.turbine_cache.cache['turbine_field_individual'][i]
        return self.power(state, tf) * self.config.site_dx(1)


class CostFunctional(GenericFunctional):
    ''' Implements a functional for computing the farm cost
    TODO: doesn't really do anything at the minute
    '''
    def __init__(self, config):
        config.turbine_cache.update(config)
        self.config = config
        # Create a copy of the parameters so that future changes will not affect the definition of this object.
        self.params = ParameterDictionary(dict(config.params))

    def cost_per_friction(self, turbines):
        #if float(self.params['cost_coef']) <= 0:
        #    return Constant(0)
        # Function spaces with polynomial degree >1 suffer from undershooting which can result in
        # negative cost values.
        if turbines.function_space().ufl_element().degree() > 1:
            raise ValueError('Costing only works if the function space for \
                    the turbine friction has polynomial degree < 2.')
        return Constant(self.params['cost_coef']) * turbines

    def Jt(self, state, tf):
        return tf * self.config.site_dx(1)#self.cost_per_friction(tf) * self.config.site_dx(1)

    def Jt_individual(self, state, i):
        ''' Computes the power output of the i'th turbine. '''
        tf = self.config.turbine_cache.cache['turbine_field_individual'][i]
        return self.cost_per_friction(tf) * self.config.site_dx(1)
        
        
class PowerCurveFunctional(GenericFunctional):
    ''' Implements a functional for the power with a given power curve
    J(u, m) = \int_\Omega power(u)
    where m controls the strength of each turbine.
    TODO: doesn't work yet...
    '''
    def __init__(self, config):
        ''' Constructs a new DefaultFunctional. The turbine settings are 
        derived from the settings params. '''
        config.turbine_cache.update(config)
        self.config = config
        # Create a copy of the parameters so that future changes will not 
        # affect the definition of this object.
        self.params = ParameterDictionary(dict(config.params))
        assert(self.params["turbine_thrust_parametrisation"] or \
               self.params["implicit_turbine_thrust_parametrisation"])

    def Jt(self, state, tf):
        up_u = state[3]  # Extract the upstream velocity
        #ux = state[0]
        def power_function(u):
            # A simple power function implementation. 
            # Could be replaced with a polynomial approximation.
            fac = Constant(1.5e6 / (3 ** 3))
            return shallow_water_model.smooth_uflmin(1.5e6, fac * u ** 3)

        P = power_function(up_u) * tf / self.config.turbine_cache.turbine_integral() * dx
        return P
