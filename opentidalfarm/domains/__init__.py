""" This module contains classes to store domain information in OpenTidalFarm. 
In particular each :class:`Domain` includes:

* the computational mesh;
* subdomains markers (used to identify turbine site areas);
* boundary markers (used to identify where boundary conditions shall be
  applied).

The mesh, and subdomain and surface makers can be visualised with 

.. code-block:: python

    plot(domain.mesh, title="Mesh")
    plot(domain.facet_ids, title="Area ids")
    plot(domain.cell_ds, title="Boundary ids")
    interactive()

"""
from file_domain import FileDomain
from rectangle_domain import RectangularDomain
