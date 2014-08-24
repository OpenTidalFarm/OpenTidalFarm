''' This test checks the correct implemetation of the turbine derivative terms.
    For that, we apply the Taylor remainder test on functional J(u, m) =
    <turbine_friction(m), turbine_friction(m)>, where m contains the turbine
    positions and the friction magnitude.
'''

from dolfin_adjoint import adj_reset
from opentidalfarm import *
from dolfin import log, INFO, ERROR

class TestTurbineDerivatives(object):
    def default_config(self):
        nx = 20
        ny = 10
        config = configuration.DefaultConfiguration(
            nx, ny, finite_element=finite_elements.p1dgp2)
        domain = domains.RectangularDomain(3000, 1000, nx, ny)
        config.set_domain(domain)
        config.params["dump_period"] = -1
        config.params["verbose"] = 0

        # Turbine settings
        config.params["turbine_pos"] = [[1000., 500.], [1600, 300], [2500, 700]]
        # The turbine friction is the control variable
        config.params["turbine_friction"] = 12.0 * numpy.random.rand(
            len(config.params["turbine_pos"]))
        config.params["turbine_x"] = 200
        config.params["turbine_y"] = 400

        return config


    def j_and_dj(self, config, m, forward_only=None):
        adj_reset()

        # Change the control variables to the config parameters
        config.params["turbine_friction"] = m[:len(
            config.params["turbine_friction"])]
        mp = m[len(config.params["turbine_friction"]):]
        config.params["turbine_pos"] = numpy.reshape(mp, (-1, 2))

        # Get initial conditions
        state = Function(config.function_space, name="current_state")
        eta0 = 2.0
        k = pi/config.domain.basin_x
        ic_expr = SinusoidalInitialCondition(config, eta0, k,
                                             config.params["depth"])
        ic = project(ic_expr, state.function_space())
        state.assign(ic, annotate=False)

        # Set the control values
        U = config.function_space.split()[0].sub(0)  # Extract the first component
                                                     # of the velocity function
                                                     # space
        U = U.collapse()  # Recompute the DOF map

        # Set up the turbine friction field using the provided control variable
        turbines = Turbines(U, config.params)
        tf = turbines()
        # The functional of interest is simply the l2 norm of the turbine field
        v = tf.vector()
        j = v.inner(v)

        if not forward_only:
            dj = []
            # Compute the derivatives with respect to the turbine friction
            for n in range(len(config.params["turbine_friction"])):
                tfd = turbines(derivative_index_selector=n,
                               derivative_var_selector='turbine_friction')
                dj.append(2 * v.inner(tfd.vector()))

            # Compute the derivatives with respect to the turbine position
            for n in range(len(config.params["turbine_pos"])):
                for var in ('turbine_pos_x', 'turbine_pos_y'):
                    tfd = turbines(derivative_index_selector=n,
                                   derivative_var_selector=var)
                    dj.append(2 * v.inner(tfd.vector()))
            dj = numpy.array(dj)

            return j, dj
        else:
            return j, None


    def test_turbine_derivatives_passes_taylor_test(self):
        # run the taylor remainder test
        config = self.default_config()
        solver = ShallowWaterSolver(config)
        m0 = ReducedFunctional(config, solver).initial_control()

        j = lambda m, forward_only = False: self.j_and_dj(config, m, forward_only)[0]
        dj = lambda m, forget: self.j_and_dj(config, m, forward_only=False)[1]

        # We set the perturbation_direction with a constant seed, so that it is
        # consistent in a parallel environment.
        p = numpy.random.rand(len(m0))

        # Run with a functional that does not depend on m directly
        minconv = helpers.test_gradient_array(j, dj, m0, 0.001, perturbation_direction=p)

        assert minconv > 1.99
