from opentidalfarm import *
from dolfin import log, INFO
import os


class TestDynamicTurbineControl(object):

    def test_gradient_passes_taylor_test(self, sw_nonlinear_problem_parameters):
        path = os.path.dirname(__file__)
        meshfile = os.path.join(path, "mesh.xml")

        domain = FileDomain(meshfile)

        config = UnsteadyConfiguration(domain)
        config.params["output_turbine_power"] = False

        sw_nonlinear_problem_parameters.finish_time = \
            sw_nonlinear_problem_parameters.start_time + \
            2 * sw_nonlinear_problem_parameters.dt

        # Deploy some turbines 
        turbine_pos = [] 
        basin_x = 640.
        basin_y = 320.
        site_x = 320.
        site_y = 160.
        site_x_start = basin_x - site_x/2
        site_y_start = basin_y - site_y/2
        config.params['turbine_x'] = 50. 
        config.params['turbine_y'] = 50. 
        config.params['controls'] = ["dynamic_turbine_friction"]
        config.params["automatic_scaling"] = False

        for x_r in numpy.linspace(site_x_start, site_x_start + site_x, 2):
            for y_r in numpy.linspace(site_y_start, site_y_start + site_y, 2):
              turbine_pos.append((float(x_r), float(y_r)))

        config.set_turbine_pos(turbine_pos, friction=1.0)
        log(INFO, "Deployed " + str(len(turbine_pos)) + " turbines.")

        config.params["turbine_friction"] = [config.params["turbine_friction"]]*3

        # Boundary conditions
        bcs = BoundaryConditionSet()
        period = 12. * 60 * 60
        eta0 = 2.0
        k = Constant(2 * pi / (period * sqrt(sw_nonlinear_problem_parameters.g * \
            sw_nonlinear_problem_parameters.depth)))
        expression = Expression(
            ("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"),
            eta0=eta0, g=sw_nonlinear_problem_parameters.g,
            depth=sw_nonlinear_problem_parameters.depth,
            t=sw_nonlinear_problem_parameters.start_time, k=k)

        bcs.add_bc("u", expression, 1, "weak_dirichlet")
        bcs.add_bc("u", expression, 2, "weak_dirichlet")
        bcs.add_bc("u", Constant((0, 0)), 3, "weak_dirichlet")

        sw_nonlinear_problem_parameters.bcs = bcs

        sw_nonlinear_problem_parameters.domain = domain

        problem = SWProblem(sw_nonlinear_problem_parameters)

        solver_params = CoupledSWSolver.default_parameters() 
        solver_params.dump_period = -1
        solver_params.cache_forward_state = False
        solver = CoupledSWSolver(problem, solver_params, config)

        rf = ReducedFunctional(config, solver, scale=10**-6)
        m0 = rf.initial_control()

        rf.j(m0)

        p = numpy.random.rand(len(m0))
        seed = 0.1
        minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=seed, perturbation_direction=p)
        assert minconv > 1.9
