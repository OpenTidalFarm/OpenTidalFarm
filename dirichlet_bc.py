from dolfin import *
class DirichletBCSet:

    def __init__(self, config):
        params = config.params
        self.config = config
        self.analytic_u = Expression(("eta0*sqrt(g*depth)*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), eta0 = params["eta0"], g = params["g"], depth = params["depth"], t = params["current_time"], k = params["k"])
        self.analytic_eta = Expression(("eta0*cos(k*x[0]-sqrt(g*depth)*k*t)", "0"), eta0 = params["eta0"], g = params["g"], depth = params["depth"], t = params["current_time"], k = params["k"])
        self.bcs = []

    def add_analytic_u(self, sub_domain):
        self.bcs.append(DirichletBC(self.config.function_space.sub(0), self.analytic_u, sub_domain))

    def add_analytic_eta(self, sub_domain):
        self.bcs.append(DirichletBC(self.config.function_space.sub(1), self.analytic_eta, sub_domain))

    def add_noslip_u(self, sub_domain):
        self.bcs.append(DirichletBC(self.config.function_space.sub(0), Constant(("0.0", "0.0")), sub_domain))

    def update_time(self, t):
        ''' Update the time values for all boundary conditions '''
        self.analytic_eta.t = t
        self.analytic_u.t = t
