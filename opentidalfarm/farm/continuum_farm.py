from dolfin import Function, FunctionSpace

class ContinuumFarm(object):
    """ Creates a farm from a mesh using the continues turbine representation.
    """
    def __init__(self, domain, site_ids=(0,), friction_function=None):

        self.domain = domain
        self.site_dx = domain.dx(site_ids)

        if friction_function is None:
            V = FunctionSpace(domain.mesh, "DG", 0)
            friction_function = Function(V)
        self.friction_func = friction_function

    @property
    def friction_function(self):
        return self.friction_func

    def update(self):
        pass
