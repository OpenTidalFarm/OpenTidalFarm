from opentidalfarm import *

class TestPowerFunctionals(object):

    def setup(self):
        prob_params = SteadySWProblem.default_parameters()

        # Create the domain
        domain = RectangularDomain(x0=0, y0=0, x1=320, y1=160, nx=160, ny=80)
        prob_params.domain = domain

        # Add a farm
        turbine = BumpTurbine(diameter=20.0, friction=12.0)
        farm = RectangularFarm(domain, site_x_start=80, site_x_end=240,
                                       site_y_start=40, site_y_end=120, turbine=turbine)
        farm.add_regular_turbine_layout(num_x=4, num_y=2)
        prob_params.tidal_farm = farm

        # Create the problem
        problem = SteadySWProblem(prob_params)

        return problem, farm

    def test_power_functional(self):

        problem, farm = self.setup()

        functional = PowerFunctional(problem)

        u0  = Constant((0, 0))
        u1  = Constant((1, 0))
        u2  = Constant((2, 0))
        u3  = Constant((3, 0))
        u4  = Constant((4, 0))

        u0_power = assemble(functional.Jt(u0, farm.friction_function))
        u1_power = assemble(functional.Jt(u1, farm.friction_function))
        u2_power = assemble(functional.Jt(u2, farm.friction_function))
        u3_power = assemble(functional.Jt(u3, farm.friction_function))
        u4_power = assemble(functional.Jt(u4, farm.friction_function))

        # Test cubic dependency on velocity
        assert 0 == u0_power
        assert abs(u2_power - 2**3*u1_power) < 1e-10
        assert abs(u4_power - 2**3*u2_power) < 1e-10


        # Test cut in/out speeds for turbines
        functional = PowerFunctional(problem, cut_in_speed=1.5, cut_out_speed=3)
        u0_power_cut = assemble(functional.Jt(u0, farm.friction_function))
        u1_power_cut = assemble(functional.Jt(u1, farm.friction_function))
        u2_power_cut = assemble(functional.Jt(u2, farm.friction_function))
        u3_power_cut = assemble(functional.Jt(u3, farm.friction_function))
        u4_power_cut = assemble(functional.Jt(u4, farm.friction_function))

        assert u0_power_cut == u0_power
        assert u1_power_cut < 1e-6      # Cut in speed kicks in
        assert u2_power_cut == u2_power
        assert u3_power_cut == u3_power
        assert u4_power_cut == u3_power # Cut out speed kicks in

