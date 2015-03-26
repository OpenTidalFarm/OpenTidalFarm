from opentidalfarm import *
import pytest

class TestCostFunctionals(object):

    def _setup(self, num_turb_x, num_turb_y):
        prob_params = SteadySWProblem.default_parameters()

        # Create the domain
        domain = RectangularDomain(x0=0, y0=0, x1=320, y1=160, nx=320, ny=160)
        prob_params.domain = domain

        # Add a farm
        turbine = BumpTurbine(diameter=20.0, friction=12.0)
        farm = RectangularFarm(domain, site_x_start=80, site_x_end=240,
                                       site_y_start=40, site_y_end=120, turbine=turbine)
        farm.add_regular_turbine_layout(num_x=num_turb_x, num_y=num_turb_y)
        prob_params.tidal_farm = farm

        # Create the problem
        problem = SteadySWProblem(prob_params)

        return problem, farm

    @pytest.mark.parametrize("n_x,n_y", [
           (1, 1),
           (2, 1),
           (2, 2),
            ])
    def test_cost_functional(self, n_x, n_y):

        problem, farm = self._setup(n_x, n_y)
        functional = CostFunctional(problem)
        state = Constant((0, 0))
        cost = assemble(functional.Jt(state, farm.friction_function))

        # Cost should be the equivalent to the integral of a single turbine
        assert abs(cost - n_x*n_y*farm.turbine_specification.integral*20*12) < 1

