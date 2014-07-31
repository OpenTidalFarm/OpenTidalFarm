''' Test description
 - single turbine
 - bubble velocity profile with maximum in the center of the domain
 - control: turbine position
 - the optimal placement for the turbine is where the velocity profile reaches its maximum (the center of the domain)
'''

from opentidalfarm import *
from opentidalfarm import helpers
from opentidalfarm.mini_model import mini_model_solve
from dolfin import log, INFO
import opentidalfarm.domains


class TestPositionOptimisation(object):
    def default_config(self):
      config = configuration.DefaultConfiguration(nx=40, ny=20, finite_element = finite_elements.p1dgp2)
      config.set_domain(opentidalfarm.domains.RectangularDomain(3000, 1000, 40, 20))
      config.params["verbose"] = 0
      config.params["dump_period"] = -1
      config.params["output_turbine_power"] = False

      # dt is used in the functional only
      config.params["dt"] = 0.8
      # Turbine settings
      # The turbine position is the control variable 
      config.params["turbine_pos"] = [[500., 200.]]
      config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
      config.params["turbine_x"] = 800
      config.params["turbine_y"] = 800
      config.params["controls"] = ['turbine_pos']
      config.params["initial_condition"] = BumpInitialCondition(config)
      config.params["automatic_scaling"] = True
      
      return config

    def test_optimisation_recovers_optimal_position(self):
        config = self.default_config()
        rf = ReducedFunctional(config, forward_model=mini_model_solve)
        m0 = rf.initial_control()

        config.info()

        p = numpy.random.rand(len(m0))
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.005, perturbation_direction=p)
        if minconv < 1.9:
          info_red("The gradient taylor remainder test failed.")
          sys.exit(1)

        # If this option does not produce any ipopt outputs, delete the ipopt.opt file
        g = lambda m: []
        dg = lambda m: []

        bounds = [[Constant(0), Constant(0)], [Constant(3000), Constant(1000)]] 

        maximize(rf, bounds = bounds, method = "SLSQP") 

        m = config.params["turbine_pos"][0]

        log(INFO, "Solution of the primal variables: m=" + repr(m) + "\n")

        assert abs(m[0]-1500) < 40
        assert abs(m[1]-500) < 0.4
