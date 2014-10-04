import os.path
from dolfin import *
from domain import Domain


class RectangularDomain(Domain):
    """ Create a rectangular domain.

    :param x0: The x coordinate of the bottom-left.
    :type x0: float
    :param y0: The y coordinate of the bottom-left.
    :type y0: float
    :param x1: The x coordinate of the top-right corner.
    :type x1: float
    :param y1: The y coordinate of the top-right corner.
    :type y1: float
    """


    def __init__(self, x0, y0, x1, y1, nx, ny):
        #: A :class:`dolfin.Mesh` containing the mesh.
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
        #: A :class:`dolfin.FacetFunction` containing the surface markers.
        self.boundaries = FacetFunction('size_t', self.mesh)
        self.boundaries.set_all(0)
        left.mark(self.boundaries, 1)
        right.mark(self.boundaries, 2)
        sides.mark(self.boundaries, 3)
        #: A :class:`dolfin.Measure` for the facet parts.
        self._ds = Measure('ds')[self.boundaries]

        #: A :class:`dolfin.CellFunction` containing the area markers.
        self.subdomains = CellFunction("size_t", self.mesh)
        self.subdomains.set_all(1)
        #: A :class:`dolfin.Measure` for the cell subdomains.
        self._dx = Measure("dx")[self.cell_ids]

        self._generate_site_dx()
        self._generate_site_vertices()
