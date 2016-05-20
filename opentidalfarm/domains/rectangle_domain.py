import os.path
import firedrake
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
        #: A :class:`firedrake.Mesh` containing the mesh.
        mpi_comm = firedrake.mpi_comm_world()
        self.mesh = firedrake.RectangleMesh(mpi_comm, firedrake.Point(x0, y0),
                firedrake.Point(x1, y1), nx, ny)

        class Left(firedrake.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and firedrake.near(x[0], x0)

        class Right(firedrake.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and firedrake.near(x[0], x1)

        class Sides(firedrake.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (firedrake.near(x[1], y0) or firedrake.near(x[1], y1))

        # Initialize sub-domain instances
        left = Left()
        right = Right()
        sides = Sides()

        # Create facet markers
        #: A :class:`firedrake.FacetFunction` containing the surface markers.
        self.facet_ids = firedrake.FacetFunction('size_t', self.mesh)
        self.facet_ids.set_all(0)
        left.mark(self.facet_ids, 1)
        right.mark(self.facet_ids, 2)
        sides.mark(self.facet_ids, 3)
        #: A :class:`firedrake.Measure` for the facet parts.
        self._ds = firedrake.Measure('ds')(subdomain_data=self.facet_ids)

        #: A :class:`firedrake.CellFunction` containing the area markers.
        self.cell_ids = firedrake.CellFunction("size_t", self.mesh)
        self.cell_ids.set_all(0)
        #: A :class:`firedrake.Measure` for the cell cell_ids.
        self._dx = firedrake.Measure("dx")(subdomain_data=self.cell_ids)
