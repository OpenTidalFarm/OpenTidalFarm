''' Test description:
 - single turbine (with constant friction distribution) whose size exceeds the size of the domain
 - constant velocity profile with an initial x-velocity of 2.
 - control: turbine friction
 - the mini model will compute a x-velocity of 2/(f + 1) wher ef is the turbine friction.
 - the functional is \int C * f * ||u||**3 where C is a constant
 - hence we maximise C * f * ( 2/(f + 1) )**3, f > 0 which has the solution f = 0.5

 Note: The solution is known only because we use a constant turbine friction distribution. 
       However this turbine model is not differentiable at its boundary, and this is why
       the turbine size has to exceed the domain.
 '''

from opentidalfarm import *

class TestCheckpoint(object):

    def default_problem(self, ic):
        domain = RectangularDomain(0, 0, 3000, 1000, 20, 10)

        # Create the farm
        farm = TidalFarm(domain)
        farm.params["turbine_pos"] = [[500., 500.]]
        # The turbine friction is the control variable 
        farm.params["turbine_friction"] = 12.0*numpy.random.rand(len(farm.params["turbine_pos"]))
        farm.params["turbine_x"] = 8000
        farm.params["turbine_y"] = 8000
        farm.params['controls'] = ['turbine_friction']
        farm.params["dump_period"] = -1
        farm.params["output_turbine_power"] = False
        farm.params["save_checkpoints"] = True
      
        # Set the problem parameters
        problem_params = DummyProblem.default_parameters()
        problem_params.domain = domain
        problem_params.finite_element = finite_elements.p1dgp2
        problem_params.initial_condition = ic(2.0, pi/3000., 50, start_time=0.0)
        problem_params.dt = 1.0  # dt is used in the functional only, 
                                 # so we set it here to 1.0
        problem_params.functional_final_time_only = True
        problem_params.tidal_farm = farm

        # Create the problem
        problem = DummyProblem(problem_params)
      
        return problem

    def test_speedup_is_larger_than_ten(self, sin_ic):
        problem = self.default_problem(ic=sin_ic)
        farm = problem.parameters.tidal_farm
        friction0 = farm.params["turbine_friction"]

        solver = DummySolver(problem)
        rf = ReducedFunctional(farm, PowerFunctional, solver,
                               automatic_scaling=False)
        bounds = [0, 100]
        
        # First optimize without checkpoints
        maxiter = 2
        t = Timer("First optimisation")
        m = maximize(rf, bounds=bounds, method="SLSQP", scale=1e-3, options={'maxiter': maxiter}) 
        t1 = t.stop()

        # Then optimize again
        farm.params["turbine_friction"] = friction0
        maxiter = 2
        t = Timer("First optimisation")
        m = maximize(rf, bounds=bounds, method="SLSQP", scale=1e-3, options={'maxiter': maxiter}) 
        t2 = t.stop()

        # Check that speedup is significant
        assert t1/t2 > 10
