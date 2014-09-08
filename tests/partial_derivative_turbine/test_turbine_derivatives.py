''' This test checks the correct implemetation of the turbine derivative terms.
    For that, we apply the Taylor remainder test on functional J(u, m) =
    <turbine_friction(m), turbine_friction(m)>, where m contains the turbine
    positions and the friction magnitude.
'''

from dolfin_adjoint import adj_reset
from opentidalfarm import *
from dolfin import log, INFO, ERROR

class TestTurbineDerivatives(object):
    def default_farm(self, domain):
        # Create turbine farm
        farm = TidalFarm(domain)
        farm.params["turbine_pos"] = [[1000., 500.], [1600, 300], [2500, 700]]
        farm.params["turbine_friction"] = 12.0 * numpy.random.rand(
            len(farm.params["turbine_pos"]))
        farm.params["turbine_x"] = 200
        farm.params["turbine_y"] = 400
        return farm


    def j_and_dj(self, problem, farm, m, forward_only=None):
        adj_reset()

        # Change the control variables to the farm parameters
        farm.params["turbine_friction"] = m[:len(farm.params["turbine_friction"])]
        mp = m[len(farm.params["turbine_friction"]):]
        farm.params["turbine_pos"] = numpy.reshape(mp, (-1, 2))

        # Set up the turbine friction field using the provided control variable
        turbines = Turbines(farm.turbine_function_space, farm.params)
        tf = turbines()
        # The functional of interest is simply the l2 norm of the turbine field
        v = tf.vector()
        j = v.inner(v)

        if not forward_only:
            dj = []
            # Compute the derivatives with respect to the turbine friction
            for n in range(len(farm.params["turbine_friction"])):
                tfd = turbines(derivative_index_selector=n,
                               derivative_var_selector='turbine_friction')
                dj.append(2 * v.inner(tfd.vector()))

            # Compute the derivatives with respect to the turbine position
            for n in range(len(farm.params["turbine_pos"])):
                for var in ('turbine_pos_x', 'turbine_pos_y'):
                    tfd = turbines(derivative_index_selector=n,
                                   derivative_var_selector=var)
                    dj.append(2 * v.inner(tfd.vector()))
            dj = numpy.array(dj)

            return j, dj
        else:
            return j, None


    def test_turbine_derivatives_passes_taylor_test(self, sin_ic):
        # run the taylor remainder test
        nx = 20
        ny = 10
        domain = RectangularDomain(0, 0, 3000, 1000, nx, ny)
        farm = self.default_farm(domain)

        eta0 = 2.0
        k = pi/3000.
        problem_params = SWProblem.default_parameters()
        problem_params.finite_element = finite_elements.p1dgp2
        problem_params.domain = domain
        problem_params.initial_condition = sin_ic(eta0, k,
                                             problem_params.depth,
                                             problem_params.start_time)
        problem_params.tidal_farm = farm
        problem = SWProblem(problem_params)

        solver_params = CoupledSWSolver.default_parameters()
        solver_params.dump_period = -1
        solver = CoupledSWSolver(problem, solver_params)

        functional = PowerFunctional
        rf_params = ReducedFunctionalParameters()
        m0 = ReducedFunctional(functional, solver, rf_params).initial_control()

        j = lambda m, forward_only = False: self.j_and_dj(problem, farm, m, forward_only)[0]
        dj = lambda m, forget: self.j_and_dj(problem, farm, m, forward_only=False)[1]

        # We set the perturbation_direction with a constant seed, so that it is
        # consistent in a parallel environment.
        p = numpy.random.rand(len(m0))

        # Run with a functional that does not depend on m directly
        minconv = helpers.test_gradient_array(j, dj, m0, 0.001, perturbation_direction=p)

        assert minconv > 1.99
