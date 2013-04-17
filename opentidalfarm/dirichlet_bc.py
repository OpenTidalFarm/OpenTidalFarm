from dolfin import *
from helpers import info, info_green, info_red, info_blue

class DirichletBCSet:

    def __init__(self, config):
        params = config.params
        self.config = config

        if params["turbine_thrust_representation"]:
            self.function_space = self.config.function_space_enriched
        else:
            self.function_space = self.config.function_space

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

        self.bcs.append(DirichletBC(self.function_space.sub(0), expression, self.config.domain.boundaries, label))

    def add_constant_flow(self, label, magnitude, direction = [1, 0]):
        norm = sqrt(direction[0]**2 + direction[1]**2)
        self.constant_inflow_bcs.append(Expression(("ux","uy"), ux=direction[0]*magnitude/norm, uy=direction[1]*magnitude/norm))
        self.bcs.append(DirichletBC(self.function_space.sub(0), self.constant_inflow_bcs[-1], self.config.domain.boundaries, label))

    def add_analytic_eta(self, label, expression):
        if self.config.params['steady_state']:
            raise ValueError, 'Can not apply a time dependent boundary condition for a steady state simulation.'
        self.expressions.append(expression)
        self.bcs.append(DirichletBC(self.function_space.sub(1), expression, self.config.domain.boundaries, label))

    def add_noslip_u(self, label):
        self.bcs.append(DirichletBC(self.function_space.sub(0), Constant(("0.0", "0.0")), self.config.domain.boundaries, label))

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
        self.bcs.append(PeriodicBC(self.function_space, pbc))

    def add_zero_eta(self, label):
        self.bcs.append(DirichletBC(self.function_space.sub(1), Constant("0.0"), self.config.domain.boundaries, label))
