from opentidalfarm import *

def _setup_config(model, flow, nx=3, ny=3, turbines=None):
    basin_x = 640.                   # some mesh metadata
    basin_y = 320.
    site_x = 320.
    site_y = 160.
    site_x_start = (basin_x - site_x)/2
    site_y_start = (basin_y - site_y)/2
    config = SteadyConfiguration(mesh, inflow_direction=[1.,0.])
    config.params["automatic_scaling"] = False
    config.set_site_dimensions(site_x_start, site_x_start + site_x,
                               site_y_start, site_y_start + site_y)

    if turbines is not None:
        config.set_turbine_pos(turbines)
    else:
        deploy_turbines(config, nx, ny)

    if model=="ApproximateShallowWater":
        model_parameters = {"wake": approx_wake}
    else:
        model_parameters = {}
    wake = AnalyticalWake(config, flow, model_type=model, model_params=model_parameters)
    rf = ReducedFunctional(config, forward_model=wake)
    return rf


def _setup(models, flows):
    results = {}
    for f in flows:
        results.update({f: {}})
    return results


def perform_tests(models, flows, nx=4, ny=2, turbines=None):
    results = _setup(models, flows)
    for flow in flows:
        for model in models:
            rf = _setup_config(model, flows[flow], nx=nx, ny=ny, turbines=turbines)
            m0 = rf.initial_control()
            info("")
            info(" Testing the %s model with %s flow ".center(80, "-") % (model, flow))
            info("")
            minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=0.5)
            descriptor = "The gradient taylor remainder test"
            info_str = "%s model with %s flow" % (model, flow)
            if minconv < 1.9 or minconv > 2.1 or numpy.isnan(minconv) or numpy.isinf(minconv):
                info_red("%s failed for the %s" % (descriptor, info_str))
                results[flow].update({model: False})
            else:
                info_green("%s passed for the %s" % (descriptor, info_str))
                results[flow].update({model: True})
    return results

def print_results(results):
    info_blue(" RESULTS ".center(40, "="))
    passes = 0
    fails = 0
    for key1 in results:
        info_str = " "+key1+" "
        info(info_str.center(40, "-"))
        for key2 in results[key1]:
            if results[key1][key2]:
                passes += 1
                info_green("%s: pass" % (key2))
            else:
                fails += 1
                info_red("%s: fail" % (key2))

    info("-".center(40,"-"))
    if fails==0:
        info_green("All %i tests passed" % passes)
    elif passes==0:
        info_red("All %i tests failed" % fails)
    else:
        info_green("Tests passed = %i" % (passes))
        info_red("Tests failed = %i" % (fails))
    info("-".center(40,"-"))


def make_wake(turbine_radius):
    import dolfin_adjoint
    # mimic a SteadyConfiguration but change a few things along the way
    config = DefaultConfiguration()
    config.params['steady_state'] = True
    config.params['initial_condition'] = ConstantFlowInitialCondition(config)
    config.params['include_advection'] = True
    config.params['include_diffusion'] = True
    config.params['diffusion_coef'] = 3.0
    config.params['quadratic_friction'] = True
    config.params['newton_solver'] = True
    config.params['friction'] = Constant(0.0025)
    config.params['theta'] = 1.0
    config.params["dump_period"] = 0

    mesh = RectangleMesh(-200, -500, 1500, 500, 50, 50)
    V, H = config.finite_element(mesh)
    config.function_space = MixedFunctionSpace([V, H])
    config.turbine_function_space = FunctionSpace(mesh, 'CG', 2)

    class Domain(object):
        """
        Domain object used for setting boundary conditions etc later on
        """
        def __init__(self, mesh):
            class InFlow(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0], -200)

            class OutFlow(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0], 1500)

            inflow = InFlow()
            outflow = OutFlow()

            self.mesh = mesh
            self.boundaries = FacetFunction("size_t", mesh)
            self.boundaries.set_all(0)
            inflow.mark(self.boundaries, 1)
            outflow.mark(self.boundaries, 2)

            self.ds = Measure('ds')[self.boundaries]

    config.domain = Domain(mesh)
    config.set_domain(config.domain, warning=False)

    # Boundary conditions
    bc = DirichletBCSet(config)
    bc.add_constant_flow(1, 1.0 + 1e-10)
    bc.add_zero_eta(2)
    config.params['bctype'] = 'strong_dirichlet'
    config.params['strong_bc'] = bc
    config.params['free_slip_on_sides'] = True

    # Optimisation settings
    config.params['functional_final_time_only'] = True
    config.params['automatic_scaling'] = False

    # Turbine settings
    config.params['turbine_pos'] = []
    config.params['turbine_friction'] = []
    config.params['turbine_x'] = turbine_radius*2
    config.params['turbine_y'] = turbine_radius*2
    config.params['controls'] = ['turbine_pos']

    # Finally set some DOLFIN optimisation flags
    parameters['form_compiler']['cpp_optimize'] = True
    parameters['form_compiler']['cpp_optimize_flags'] = '-O3'
    parameters['form_compiler']['optimize'] = True

    # place a turbine with default friction
    turbine = [(0., 0.)]
    config.set_turbine_pos(turbine)

    # solve the shallow water equations
    rf = ReducedFunctional(config)
    info_blue("Generating the wake model...")
    rf.j(rf.initial_control())
    # get state
    state = rf.last_state
    V = VectorFunctionSpace(config.function_space.mesh(),
                                   "CG", 1, dim=2)
    u_out = TrialFunction(V)
    v_out = TestFunction(V)
    M_out = assemble(inner(v_out, u_out)*dx)
    out_state = Function(V)
    rhs = dolfin_adjoint.assemble(inner(v_out, state.split()[0]) *dx)
    dolfin_adjoint.solve(M_out, out_state.vector(), rhs, "cg", "sor", annotate=False)
    info_green("Wake model generated.")
    return out_state



#
# run the convergence tests
#
if __name__=='__main__':
    # define a mesh and some flow models to use
    mesh = "mesh.xml" # choose a mesh
    V = VectorFunctionSpace(Mesh(mesh), "CG", 2)
    flow_const = interpolate(Expression(("2.0", "0.0")), V)
    flow_var_xx = interpolate(Expression(("2.0+x[0]/1280.", "0.0")), V)
    flow_var_xy = interpolate(Expression(("2.0+x[1]/1280.", "0.0")), V)
    flow_var_yy = interpolate(Expression(("0.0", "2.0+x[1]/640.")), V)
    flow_var_yx = interpolate(Expression(("0.0", "2.0+x[0]/640.")), V)
    flow_var = interpolate(Expression(("1.0+x[0]/1280.", "1.0+x[1]/640.")), V)

    # flow dictionary
    flows = {"constant": flow_const,
             "varying x due to x": flow_var_xx,
             "varying x due to y": flow_var_xy,
             "varying y due to x": flow_var_yx,
             "varying y due to y": flow_var_yy,
             "fully varying": flow_var
             }

    # wake models to test
    #models = ["Jensen", "Simple"]
    models = ["ApproximateShallowWater"]
    # make some wake so it doesn't have to be done each time a model is
    # initialised
    approx_wake = make_wake(10)


    # normally deploy turbines
    normal_results = perform_tests(models, flows, 2, 2)
    # randomly deployed turbines
    import numpy
    turbines = numpy.random.rand(9, 2)*160
    for i in range(len(turbines)):
        if i%2==0:
            turbines[i] += 160.
        else:
            turbines[i] += 80.
    random_results = perform_tests(models, flows, turbines=turbines)

    info("")
    info_blue(" Normally deployed turbines ".center(40, "="))
    print_results(normal_results)
    info("")
    info_blue(" Randomly deployed turbines ".center(40, "="))
    print_results(random_results)
    info("")
