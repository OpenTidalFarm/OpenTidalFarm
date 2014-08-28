import os.path
from dolfin import *
from domain import Domain


class RectangularDomain(Domain):
    """ Create a rectangular domain. 

    :param x0: The x coordinate of the bottom-left.
    :type x0: float.
    :param y0: The y coordinate of the bottom-left.
    :type y0: float.
    :param x1: The x coordinate of the top-right corner.
    :type x1: float.
    :param y1: The y coordinate of the top-right corner.
    :type y1: float.
    """


    def __init__(self, x0, y0, x1, y1, nx, ny):
        self.mesh = RectangleMesh(x0, x0, x1, y1, nx, ny)

        class Left(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], x0)

        class Right(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], x1)

        class Sides(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (near(x[1], y0) or near(x[1], y1))

        # Initialize sub-domain instances
        left = Left()
        right = Right()
        sides = Sides()

        # Create facet markers
        self.facet_ids = FacetFunction('size_t', self.mesh)
        self.facet_ids.set_all(0)
        left.mark(self.facet_ids, 1)
        right.mark(self.facet_ids, 2)
        sides.mark(self.facet_ids, 3)
        self.ds = Measure('ds')[self.facet_ids]

        # Read cell markers
        self.cell_ids = CellFunction("size_t", self.mesh)
        self.cell_ids.set_all(1)
        self.dx = Measure("dx")[self.cell_ids]  
