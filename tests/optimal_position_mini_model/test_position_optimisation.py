''' Test description
 - single turbine
 - bubble velocity profile with maximum in the center of the domain
 - control: turbine position
 - the optimal placement for the turbine is where the velocity profile reaches
   its maximum (the center of the domain)
'''

from opentidalfarm import *
from opentidalfarm import helpers
from dolfin import log, INFO


def BumpInitialCondition(x0, y0, x1, y1):

    class BumpExpr(Expression):
        '''This class implements a initial condition with a bump velocity profile.
           With that we know that the optimal turbine location must be in the center of the domain. '''

        def bump_function(self, x):
            '''The velocity is initially a bump function (a smooth function with limited support):
                       /  e**-1/(1-x**2)   for |x| < 1
              psi(x) = |
                       \  0   otherwise
              For more information see http://en.wikipedia.org/wiki/Bump_function
            '''
            if x[0] ** 2 < 1 and x[1] ** 2 < 1:
                bump = exp(-1.0 / (1.0 - x[0] ** 2))
                bump *= exp(-1.0 / (1.0 - x[1] ** 2))
                bump /= exp(-1) ** 2
            else:
                bump = 0.0
            return bump

        def eval(self, values, X):
            x_unit = 2 * (x1 - X[0]) / (x1-x0) - 1.0
            y_unit = 2 * (y1 - X[1]) / (y1-y0) - 1.0

            values[0] = self.bump_function([x_unit, y_unit])
            values[1] = 0
            values[2] = 0

        def value_shape(self):
            return (3,)

    return BumpExpr()

class TestPositionOptimisation(object):
    def default_config(self):
        domain = RectangularDomain(0, 0, 3000, 1000, 40, 20)

        config = DefaultConfiguration(domain)
        config.params["verbose"] = 0
  
        problem_params = DummyProblem.default_parameters()
  
        # dt is used in the functional only
        problem_params.dt = 0.8
        problem_params.functional_final_time_only = False
        problem_params.finite_element = finite_elements.p1dgp2
        problem_params.domain = domain
        problem_params.initial_condition = BumpInitialCondition(0, 0, 3000, 1000)

        # Turbine settings
        # The turbine position is the control variable 
        config.params["turbine_pos"] = [[500., 200.]]
        config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
        config.params["turbine_x"] = 800
        config.params["turbine_y"] = 800
        config.params["controls"] = ['turbine_pos']
        
        problem = DummyProblem(problem_params)
  
        return problem, config

    def test_optimisation_recovers_optimal_position(self):
        problem, config = self.default_config()

        solver = DummySolver(problem, config)
        functional = PowerFunctional
        rf = ReducedFunctional(config, functional, solver, automatic_scaling=5.)
        m0 = rf.initial_control()

        config.info()

        p = numpy.random.rand(len(m0))
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.005, 
                                              perturbation_direction=p)
        assert minconv > 1.9

        bounds = [[Constant(0), Constant(0)], [Constant(3000), Constant(1000)]] 
        maximize(rf, bounds=bounds, method="SLSQP") 

        m = config.params["turbine_pos"][0]
        log(INFO, "Solution of the primal variables: m=" + repr(m) + "\n")

        assert abs(m[0]-1500) < 40
        assert abs(m[1]-500) < 0.4
