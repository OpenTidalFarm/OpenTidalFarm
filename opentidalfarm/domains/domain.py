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
