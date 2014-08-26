import os.path
from dolfin import *
from domain import Domain


class FileDomain(Domain):
    ''' Loads a mesh from external files. '''

    def __init__(self, mesh_xml, facet_ids_xml=None, cell_ids_xml=None):
        """ Initialises a new mesh domain from external files. 

        :param mesh_xml: The .xml file of the mesh.
        :type mesh_xml: str.
        :param facet_ids_xml: The .xml file containing the facet ids of the mesh. 
            If None, this defaults to :param:`mesh_xml` + "_facet_region.xml".
        :type facet_ids_xml: str.
        :param cell_ids_xml: The .xml file containing the cell ids of the mesh. 
            If None, this defaults to :param:`mesh_xml` + "_physical_region.xml".
        :type cell_ids_xml: str.
        """

        self.mesh = Mesh(mesh_xml)

        # Read facet markers
        if facet_ids_xml is None:
            facet_ids_xml = os.path.splitext(mesh_xml)[0] + "_facet_region.xml"

        self.facet_ids = MeshFunction('size_t', self.mesh, facet_ids_xml)
        self.ds = Measure('ds')[self.facet_ids]

        # Read cell markers
        if cell_ids_xml is None:
            cell_ids_xml = os.path.splitext(mesh_xml)[0] + "_physical_region.xml"

        self.cell_ids = MeshFunction("size_t", self.mesh, cell_ids_xml)
        self.dx = Measure("dx")[self.cell_ids]  
