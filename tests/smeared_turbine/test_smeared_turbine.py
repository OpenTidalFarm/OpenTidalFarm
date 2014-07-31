''' This test checks the correctness of the gradient with the
    smeared turbine representation.
'''
from opentidalfarm import *


class TestSmearedTurbine(object):

    def test_gradient_passes_taylor_test(self):
        parameters["form_compiler"]["quadrature_degree"] = 4

        nx = 5
        ny = 5
        config = DefaultConfiguration(nx, ny)
        domain = domains.RectangularDomain(3000, 1000, nx, ny)
        config.set_domain(domain)
        config.params['finish_time'] = config.params["start_time"] + \
            3*config.params["dt"]

        # Switch to a smeared turbine representation
        config.params["controls"] = ["turbine_friction"]
        config.params["turbine_parametrisation"] = "smeared"
        config.params["dump_period"] = -1
        config.params["output_turbine_power"] = False
        config.params['initial_condition'] = ConstantFlowInitialCondition(
            config,
            val=[1, 0, 0]
        )

        site_x_start = 750
        site_x = 1500
        site_y_start = 250
        site_y = 500

        k = pi/site_x
        config.params["flather_bc_expr"] = Expression(
            ("2*eta0*sqrt(g/depth)*cos(-sqrt(g*depth)*k*t)", "0"),
            eta0=2.,
            g=config.params["g"],
            depth=config.params["depth"],
            t=config.params["current_time"],
            k=k
        )


        class Site(SubDomain):
            def inside(self, x, on_boundary):
                return (between(x[0], (site_x_start, site_x_start+site_x)) and
                        between(x[1], (site_y_start, site_y_start+site_y)))

        site = Site()
        d = CellFunction("size_t", config.domain.mesh)
        d.set_all(0)
        site.mark(d, 1)
        config.site_dx = Measure("dx")[d]

        rf = ReducedFunctional(config)
        # Ensure the same seed value accross all CPUs
        numpy.random.seed(33)
        m0 = numpy.random.rand(len(rf.initial_control()))

        seed = 0.1
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=seed)

        assert minconv > 1.9
