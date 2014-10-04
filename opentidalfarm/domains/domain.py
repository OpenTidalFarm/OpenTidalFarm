from dolfin import *


class Domain(object):
    """ An abstract domain class. """

    def __init__(self):
        raise NotImplementedError("Domain is a base class only.")

    def __str__(self):
        comm = mpi_comm_world()
        hmin = MPI.min(comm, self.mesh.hmin())
        hmax = MPI.max(comm, self.mesh.hmax())
        num_cells = MPI.sum(comm, self.mesh.num_cells())

        s = "Number of mesh elements: %i.\n" % num_cells
        s += "Mesh element size: %f - %f." % (hmin, hmax)

        return s

    @property
    def ds(self):
        """A :class:`dolfin.Measure` for the facet parts of the domain."""
        return self._ds


    @property
    def dx(self):
        """A :class:`dolfin.Measure` for the cell subdomains."""
        return self._dx


    @property
    def site_dx(self):
        """A :class:`dolfin.Measure` for the turbine site."""
        # TODO: some explanation for smeared approach.
        return self._site_dx


    def _generate_site_dx(self):
        # Define the subdomain for the turbine site. The default value should
        # only be changed for smeared turbine representations.
        domains = CellFunction("size_t", self.mesh)
        domains.set_all(1)
        # The measure used to integrate the turbine friction.
        self._site_dx = Measure("dx")[domains]


    def _generate_site_vertices(self):
        # Extract the submesh for the site.
        self._site_mesh = dolfin.SubMesh(self.mesh, self.subdomains, 1)

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
        self.site_vertices = site_mesh_coordinates[self._boundary_indices]
