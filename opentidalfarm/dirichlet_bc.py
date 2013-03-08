from dolfin import *
from helpers import info, info_green, info_red, info_blue

def ConstantFlowBoundaryCondition(config, direction):
    ''' Specifies a constant flow in direction "direction". '''
    class ConstantFlow(Expression):
        def __init__(self):
            self.t = config.params["start_time"]
            self.depth = config.params["depth"]
            self.k = config.params["k"]
            self.g = config.params["g"]
            self.eta0 = config.params["eta0"]

        def l2norm(self, x):
            return sqrt(x[0]**2 + x[1]**2)

        def eval(self, values, X):
            values[0] = direction[0]/self.l2norm(direction) * self.eta0 * sqrt(self.g / self.depth) 
            values[1] = direction[1]/self.l2norm(direction) * self.eta0 * sqrt(self.g / self.depth) 
        def value_shape(self):
            return (2,)

    return ConstantFlow 

class DirichletBCSet:

    def __init__(self, config):
        params = config.params
        self.config = config

        #self.analytic_u = Expression(("eta0*sqrt(g/depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), eta0 = params["eta0"], g = params["g"], depth = params["depth"], t = params["current_time"], k = params["k"])
        #self.analytic_eta = Expression("eta0*cos(k*x[0]-sqrt(g*depth)*k*t)", eta0 = params["eta0"], g = params["g"], depth = params["depth"], t = params["current_time"], k = params["k"])
        self.expressions = []
        self.constant_inflow_bcs = []

        self.bcs = []

    def update_time(self, t):
        ''' Update the time values for all boundary conditions '''
        for expression in self.expressions:
            expression.t = t
        for bc in self.constant_inflow_bcs:
            bc.t = t

    def add_analytic_u(self, label, expression):
        if self.config.params['steady_state']:
            raise ValueError, 'Can not apply a time dependent boundary condition for a steady state simulation.'
        self.expressions.append(expression)
        self.bcs.append(DirichletBC(self.config.function_space.sub(0), expression, self.config.domain.boundaries, label))

    def add_constant_flow(self, label, direction = [1, 0]):
        self.constant_inflow_bcs.append(ConstantFlowBoundaryCondition(self.config, direction)())
        self.bcs.append(DirichletBC(self.config.function_space.sub(0), self.constant_inflow_bcs[-1], self.config.domain.boundaries, label))

    def add_analytic_eta(self, label, expression):
        if self.config.params['steady_state']:
            raise ValueError, 'Can not apply a time dependent boundary condition for a steady state simulation.'
        self.expressions.append(expression)
        self.bcs.append(DirichletBC(self.config.function_space.sub(1), expression, self.config.domain.boundaries, label))

    def add_noslip_u(self, label):
        self.bcs.append(DirichletBC(self.config.function_space.sub(0), Constant(("0.0", "0.0")), self.config.domain.boundaries, label))

    def add_periodic_sides(self):
        config = self.config

        # Sub domain for Periodic boundary condition
        class PeriodicBoundary(SubDomain):

            def inside(self, x, on_boundary):
                return bool(x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary)

            def map(self, x, y):
                y[1] = x[1] - config.domain.basin_y 
                y[0] = x[0]

        pbc = PeriodicBoundary()
        self.bcs.append(PeriodicBC(config.function_space, pbc))

    def add_zero_eta(self, label):
        self.bcs.append(DirichletBC(self.config.function_space.sub(1), Constant("0.0"), self.config.domain.boundaries, label))
