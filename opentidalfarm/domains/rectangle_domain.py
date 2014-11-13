import os.path
import dolfin
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
        self.mesh = dolfin.RectangleMesh(x0, x0, x1, y1, nx, ny)

        class Left(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and dolfin.near(x[0], x0)

        class Right(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and dolfin.near(x[0], x1)

        class Sides(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (dolfin.near(x[1], y0) or dolfin.near(x[1], y1))

        # Initialize sub-domain instances
        left = Left()
        right = Right()
        sides = Sides()

        # Create facet markers
        #: A :class:`dolfin.FacetFunction` containing the surface markers.
        self.facet_ids = dolfin.FacetFunction('size_t', self.mesh)
        self.facet_ids.set_all(0)
        left.mark(self.facet_ids, 1)
        right.mark(self.facet_ids, 2)
        sides.mark(self.facet_ids, 3)
        #: A :class:`dolfin.Measure` for the facet parts.
        self._ds = dolfin.Measure('ds')[self.facet_ids]

        #: A :class:`dolfin.CellFunction` containing the area markers.
        self.cell_ids = dolfin.CellFunction("size_t", self.mesh)
        self.cell_ids.set_all(0)
        #: A :class:`dolfin.Measure` for the cell cell_ids.
        self._dx = dolfin.Measure("dx")[self.cell_ids]
