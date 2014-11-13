import os.path
import dolfin
from domain import Domain


class FileDomain(Domain):
    """ Create a domain from DOLFIN mesh files (.xml).

    :param mesh_file: The .xml file of the mesh.
    :type mesh_file: str
    :param facet_ids_file: The .xml file containing the facet ids of the mesh.
        If None, the default is to `mesh_file` + "_facet_region.xml".
    :type facet_ids_file: str
    :param cell_ids_file: The .xml file containing the cell ids of the mesh.
        If None, the default is to `mesh_file` + "_physical_region.xml".
    :type cell_ids_file: str
    """

    def __init__(self, mesh_file, facet_ids_file=None, cell_ids_file=None):

        #: A :class:`dolfin.Mesh` containing the mesh.
        self.mesh = dolfin.Mesh(mesh_file)

        # Read facet markers
        if facet_ids_file is None:
            facet_ids_file = (os.path.splitext(mesh_file)[0] +
                              "_facet_region.xml")

        # Read cell markers
        if cell_ids_file is None:
            cell_ids_file = (os.path.splitext(mesh_file)[0] +
                            "_physical_region.xml")

        #: A :class:`dolfin.FacetFunction` containing the surface markers.
        self.facet_ids = dolfin.MeshFunction("size_t", self.mesh, facet_ids_file)
        #: A :class:`dolfin.Measure` for the facet parts.
        self._ds = dolfin.Measure('ds')[self.facet_ids]

        #: A :class:`dolfin.CellFunction` containing the area markers.
        self.cell_ids = dolfin.MeshFunction("size_t", self.mesh, cell_ids_file)
        #: A :class:`dolfin.Measure` for the cell subdomains.
        self._dx = dolfin.Measure("dx")[self.cell_ids]
