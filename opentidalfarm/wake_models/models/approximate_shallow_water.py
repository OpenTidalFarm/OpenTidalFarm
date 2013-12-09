from model import Model
from ..helpers import ADDolfinExpression
import dolfin
import dolfin_adjoint
import opentidalfarm as otf

class ApproximateShallowWater(Model):
    """
    Approximates the shallow water equations by using the wake of a single
    turbine in 1m/s flow as a wake model.
    """
    def __init__(self, flow_field, turbine_radius, model_parameters=None):
        # add radius to model params
        if model_parameters is None:
            model_parameters = {"radius": turbine_radius}
        else:
            model_parameters.update({"radius": turbine_radius})
        # check if gradient required
        if "compute_gradient" in model_parameters:
            compute_gradient = model_parameters["compute_gradient"]
        else:
            compute_gradient = True
        # check if a wake and wake gradients are provided
        if "wake" in model_parameters:
            if "wake_gradients" in model_parameters:
                wake_gradients = model_parameters["wake_gradients"]
            else:
                wake_gradients = None
            self.wake = ADDolfinExpression(model_parameters["wake"],
                                           compute_gradient=compute_gradient,
                                           gradients=wake_gradients)
        # compute wake and gradients
        else:
            # mimic a SteadyConfiguration but change a few things along the way
            config = otf.DefaultConfiguration()
            config.params['steady_state'] = True
            config.params['initial_condition'] = otf.ConstantFlowInitialCondition(config)
            config.params['include_advection'] = True
            config.params['include_diffusion'] = True
            config.params['diffusion_coef'] = 3.0
            config.params['quadratic_friction'] = True
            config.params['newton_solver'] = True
            config.params['friction'] = dolfin.Constant(0.0025)
            config.params['theta'] = 1.0
            config.params["dump_period"] = 0

            xmin, ymin = -100, -200
            xsize, ysize = 1000, 400
            xcells, ycells = 400, 160
            mesh = dolfin.RectangleMesh(xmin, ymin, xmin+xsize, ymin+ysize,
                                        xcells, ycells)

            V, H = config.finite_element(mesh)
            config.function_space = dolfin.MixedFunctionSpace([V, H])
            config.turbine_function_space = dolfin.FunctionSpace(mesh, 'CG', 2)

            class Domain(object):
                """
                Domain object used for setting boundary conditions etc later on
                """
                def __init__(self, mesh):
                    class InFlow(dolfin.SubDomain):
                        def inside(self, x, on_boundary):
                            return dolfin.near(x[0], -100)

                    class OutFlow(dolfin.SubDomain):
                        def inside(self, x, on_boundary):
                            return dolfin.near(x[0], 900)

                    inflow = InFlow()
                    outflow = OutFlow()

                    self.mesh = mesh
                    self.boundaries = dolfin.FacetFunction("size_t", mesh)
                    self.boundaries.set_all(0)
                    inflow.mark(self.boundaries, 1)
                    outflow.mark(self.boundaries, 2)

                    self.ds = dolfin.Measure('ds')[self.boundaries]

            config.domain = Domain(mesh)
            config.set_domain(config.domain, warning=False)

            # Boundary conditions
            bc = otf.DirichletBCSet(config)
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
            dolfin.parameters['form_compiler']['cpp_optimize'] = True
            dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'
            dolfin.parameters['form_compiler']['optimize'] = True

            # place a turbine with default friction
            turbine = [(0., 0.)]
            config.set_turbine_pos(turbine)

            # solve the shallow water equations
            rf = otf.ReducedFunctional(config, plot=False)
            dolfin.info_blue("Generating the wake model...")
            rf.j(rf.initial_control())
            # get state
            state = rf.last_state
            V = dolfin.VectorFunctionSpace(config.function_space.mesh(),
                                           "CG", 2, dim=2)
            u_out = dolfin.TrialFunction(V)
            v_out = dolfin.TestFunction(V)
            M_out = dolfin_adjoint.assemble(dolfin.inner(v_out, u_out)*dolfin.dx)
            out_state = dolfin.Function(V)
            rhs = dolfin_adjoint.assemble(dolfin.inner(v_out, state.split()[0])
                                          *dolfin.dx)
            dolfin_adjoint.solve(M_out, out_state.vector(), rhs, "cg", "sor",
                                 annotate=False)
            dolfin.info_green("Wake model generated.")
            self.wake = ADDolfinExpression(out_state.split()[0],
                                           compute_gradient)

        super(ApproximateShallowWater, self).__init__("ApproximateShallowWater",
                                                      flow_field,
                                                      model_parameters)


    def wake_radius(self, x0):
        """
        Fixed value
        """
        return 200.


    def _upstream_wake(self):
        """
        Fixed value
        """
        return 100.


    def individual_factor(self, x0, y0):
        """
        Returns the individual velocity reduction factor
        """
        return self.wake((x0, y0))


    def get_search_radius(self, recovery_loss=None):
        """
        Fixed value
        """
        return 900.
