''' Test description:
 - single turbine (with constant friction distribution) whose size exceeds the size of the domain
 - constant velocity profile with an initial x-velocity of 2.
 - control: turbine friction
 - the mini model will compute a x-velocity of 2/(f + 1) where f is the turbine friction.
 - the functional is \int C * f * ||u||**3 where C is a constant
 - hence we maximise C * f * ( 2/(f + 1) )**3, f > 0 which has the solution f = 0.5

 Note: The solution is known only because we use a constant turbine friction distribution. 
       However this turbine model is not differentiable at its boundary, and this is why
       the turbine size has to exceed the domain.
 '''

from opentidalfarm import *

class TestFrictionOptimisation(object):

    def default_config(self):
      domain = RectangularDomain(0, 0, 3000, 1000, 20, 10)

      config = DefaultConfiguration(domain)
      config.params["verbose"] = 0

      problem_params = DummyProblem.default_parameters()

      # dt is used in the functional only, so we set it here to 1.0
      problem_params.dt = 1.0
      problem_params.functional_final_time_only = True

      # Turbine settings
      config.params["turbine_pos"] = [[500., 500.]]
      # The turbine friction is the control variable 
      config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
      config.params["turbine_x"] = 1e10
      config.params["turbine_y"] = 1e10
      config.params['controls'] = ['turbine_friction']
      config.params["output_turbine_power"] = False

      k = pi/3000.
      config.params['k'] = k
      problem_params.initial_condition = SinusoidalInitialCondition(2.0, k,
              50., 0.0)

      problem_params.domain = domain
      problem_params.finite_element = finite_elements.p1dgp2
      problem = DummyProblem(problem_params)
      return problem, config

    def test_optimisation_recovers_optimal_friction(self):

        problem, config = self.default_config()
        solver = DummySolver(problem, config)

        rf = ReducedFunctional(config, solver, scale=1e-3)
        m0 = rf.initial_control()
        rf(m0)
        rf.dj(m0, forget=False)

        p = numpy.random.rand(len(m0))
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.001, perturbation_direction=p)
        assert minconv > 1.98

        bounds = [0, 100]
        maximize(rf, bounds=bounds, method="SLSQP", scale=1e-3) 

        assert abs(config.params["turbine_friction"][0] - 0.5) < 10**-4
