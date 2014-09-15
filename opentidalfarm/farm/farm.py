import os
import dolfin
from .base_farm import BaseFarm


class Farm(BaseFarm):
    """Extends :py:class:`BaseFarm`. Creates a farm from a mesh.

    This class holds the turbines within a site defined by a mesh with a turbine
    site marked by `1`.

    """
    def __init__(self, mesh):
        """Initializes an empty farm defined by the mesh.

        :param mesh: The name of the mesh file to use, e.g. 'mesh.xml' if the
            file is located at `./mesh.xml`.
        :type mesh: string

        """
        # Initialize the base class
        super(Farm, self).__init__()

        try:
            assert(mesh.endswith(".xml"))
        except AssertionError:
            raise ValueError("The mesh file (%s) should end in '.xml'" % mesh)

        # Get the basename.
        mesh_basename = mesh.split(".xml")[0]
        facet = mesh_basename+"_facet_region.xml"
        physical = mesh_basename+"_physical_region.xml"

        def check_exists(filename):
            try:
                assert(os.path.isfile(filename))
            except AssertionError:
                raise ValueError("%s does not exist or is not a file.")

        # Check the three files we need exist.
        check_exists(mesh)
        check_exists(facet)
        check_exists(physical)

        # Load the mesh, boundaries and subdomains.
        self._mesh = dolfin.Mesh(mesh_basename+".xml")
        self._boundaries = dolfin.MeshFunction("size_t", self._mesh, facet)
        self._subdomains = dolfin.MeshFunction("size_t", self._mesh, physical)

        # Extract the submesh for the site.
        self._site_mesh = dolfin.SubMesh(self._mesh, self._subdomains, 1)

        # Create measures over the subdomains.
        self._ds = dolfin.Measure("ds")[self._boundaries]
        self._dx = dolfin.Measure("dx")[self._subdomains]

        # Mark a CG1 Function with ones on the boundary.
        V = dolfin.FunctionSpace(self._site_mesh, 'CG', 1)
        bc = dolfin.DirichletBC(V, 1, dolfin.DomainBoundary())
        u = dolfin.Function(V)
        bc.apply(u.vector())

        # Get vertices sitting on boundary.
        d2v = dolfin.dof_to_vertex_map(V)
        self._boundary_indices = d2v[u.vector()==1]

        # Get the vertex coordinates on the boundary.
        site_mesh_coordinates = self._site_mesh.coordinates()
        self._site_vertices = site_mesh_coordinates[self._boundary_indices]


    @property
    def ds(self):
        """A dolfin.Measure along the boundaries."""
        return self._ds


    @property
    def dx(self):
        """A dolfin.Measure for the surface of the domain."""
        return self._dx
