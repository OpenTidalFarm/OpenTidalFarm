''' This tests the upwind-correction in the ThrustTurbine
which compensates for the fact that the thrust coefficient
is defined wrt to the upstream. The test checks that the
correct force is applied to flow.
'''
from opentidalfarm import *
import os.path
from math import pi


class TestTurbineCorrection(object):

    def test_turbine_correction(self, steady_sw_problem_parameters):
        parameters["form_compiler"]["quadrature_degree"] = 4
        prob_params=steady_sw_problem_parameters

        # Load domain
        path = os.path.dirname(__file__)
        meshfile = os.path.join(path, "mesh_coarse.xml")
        domain = FileDomain(meshfile)

        C_t = 0.6
        turbine = ThrustTurbine(diameter=20.,
                c_t_design=C_t,
                cut_in_speed=1.,
                cut_out_speed=5.0,
                water_depth=prob_params.depth)

        # Boundary conditions
        site_x_start = 160.
        site_x = 320.
        site_y_start = 140.
        site_y = 40.

        farm = RectangularFarm(domain,
                               site_x_start=site_x_start,
                               site_x_end=site_x_start+site_x,
                               site_y_start=site_y_start,
                               site_y_end=site_y_start+site_y,
                               turbine=turbine)
        farm.add_turbine((180.,160.))
        prob_params.tidal_farm = farm

        prob_params.domain = domain
        prob_params.initial_condition = Constant((3, 0, 0))

        bcs = BoundaryConditionSet()
        bcs.add_bc("u", Constant((3.0, 0.0)), facet_id=1, bctype="strong_dirichlet")
        bcs.add_bc("eta", Constant(0.0), facet_id=2, bctype="weak_dirichlet")
        bcs.add_bc("u", facet_id=3, bctype="free_slip")
        prob_params.bcs = bcs

        class Site(SubDomain):
            def inside(self, x, on_boundary):
                return (between(x[0], (site_x_start, site_x_start+site_x)) and
                        between(x[1], (site_y_start, site_y_start+site_y)))

        site = Site()
        d = CellFunction("size_t", farm.domain.mesh)
        d.set_all(0)
        site.mark(d, 1)
        farm.site_dx = Measure("dx")(subdomain_data=d)

        problem = SteadySWProblem(prob_params)

        solver_params = CoupledSWSolver.default_parameters()
        solver_params.dolfin_solver["newton_solver"]["relative_tolerance"] = 1e-7
        solver = CoupledSWSolver(problem, solver_params)

        for s in solver.solve():
            pass

        tf = s['tf']
        u = s['u']
        scaling = farm.turbine_specification.turbine_parametrisation_constant
        F_applied = assemble(tf*scaling*C_t*dot(u,u)*dx)
        F_desired = 0.5*0.6*pi*100.*9
        print("F_applied, F_desired: ", F_applied, F_desired)
        err = abs((F_applied-F_desired)/F_desired)
        print("rel. error = ", err)
        assert(err < 0.5/100.)
