''' This tests the upwind-correction in the ThrustTurbine
which compensates for the fact that the thrust coefficient
is defined wrt to the upstream. The test checks that the
correct force is applied to flow.
'''
from opentidalfarm import *
import os.path
from math import pi
import numpy
import scipy.interpolate


class TestTurbineCorrection(object):

    def test_thrust_curve(self, steady_sw_problem_parameters):
        parameters["form_compiler"]["quadrature_degree"] = 4
        prob_params=steady_sw_problem_parameters

        # Load domain
        path = os.path.dirname(__file__)
        meshfile = os.path.join(path, "mesh_coarse.xml")
        domain = FileDomain(meshfile)

        C_t = 0.6
        thrust_curve = standard_thrust_curve()
        thrust_curve = numpy.array(thrust_curve)
        thrust_interpolator = scipy.interpolate.interp1d(thrust_curve[:,0], thrust_curve[:,1])

        D = 20 # turbine diameter
        turbine = BumpTurbine(diameter=D,
                thrust_curve=thrust_curve, depth=prob_params.depth)

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
        u_constant = Constant((0.0, 0.0))

        bcs = BoundaryConditionSet()
        bcs.add_bc("u", u_constant, facet_id=1, bctype="strong_dirichlet")
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

        results = []
        for u_in in numpy.linspace(0,4.99,7):
            print "u_in = ", u_in
            u_constant.assign(Constant((u_in, 0.0)))
            try:
                C_t = thrust_interpolator(u_in)
            except ValueError:
                C_t = 0.0 # value outside the thrust_curve range, below cut_in or above cut_out
            for s in solver.solve(annotate=False):
                pass

            tf = s['tf']
            u = s['u']
            F_vec = farm.force(u)
            F_applied = assemble(sqrt(dot(F_vec, F_vec))*farm.site_dx)
            # this computes the desired force based on the upwind speed: F=0.5*C_t*A_t*u**2
            F_desired = 0.5 * C_t * pi*(D/2)**2 * u_in**2
            print "F_applied, F_desired: ", F_applied, F_desired
            err = abs((F_applied-F_desired)/(F_desired+1e-12))
            print "rel. error = ", err

            u_mag = sqrt(dot(u, u))
            P_computed = assemble(farm.power_integral(u_mag))
            C_P = C_t * (1+sqrt(1-C_t))/2.0
            P_desired = 0.5 * C_P * pi*(D/2)**2 * u_in**3
            print "P_computed, P_desired: ", P_computed, P_desired
            err = abs((P_computed-P_desired)/(P_desired+1e-12))
            print "rel. error = ", err

            results.append([u_in, F_applied, F_desired, P_computed, P_desired])

        results = numpy.array(results)
        l2_error_force = numpy.linalg.norm(results[:,1]-results[:,2])/sqrt(results.shape[0])
        print "l2 error in force: ", l2_error_force
        assert l2_error_force < results[:,2].max()/100.
        l2_error_power = numpy.linalg.norm(results[:,3]-results[:,4])/sqrt(results.shape[0])
        print "l2 error in power: ", l2_error_power
        assert l2_error_power < results[:,4].max()/100.

        return
        import pylab
        pylab.figure()
        pylab.plot(results[:,0], results[:,1])
        pylab.plot(results[:,0], results[:,2])
        pylab.figure()
        pylab.plot(results[:,0], results[:,3])
        pylab.plot(results[:,0], results[:,4])
        pylab.show()
