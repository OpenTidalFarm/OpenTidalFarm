Installation
============

OpenTidalFarm runs on Windows, Mac OSX and Linux.

We recommend to install OpenTidalFarm using Docker:

Docker images (all platforms and versions)
------------------------------------------

`Docker <https://www.docker.com>`_ allows us to build and ship
consistent high-performance OpenTidalFarm installations for almost any
platform. To get started, follow these steps:

#. Install Docker. Mac and Windows users should install the `Docker
   Toolbox <https://www.docker.com/products/docker-toolbox>`_ (this is
   a simple one-click install) and Linux users should `follow these
   instructions <https://docs.docker.com/linux/step_one/>`_.
#. Install the FEniCS Docker script::

    curl -s https://get.fenicsproject.org | bash

   If running on Mac or Windows, make sure you run this and other
   commands inside the Docker Quickstart Terminal.

#. Start an OpenTidalFarm session by running the following command::

    fenicsproject run quay.io/opentidalfarm/virtual

   A Jupyter notebook instance with a user defined name (here myproject) can be started with::

    fenicsproject notebook myproject quay.io/opentidalfarm/virtual
    fenicsproject start myproject


   The FEniCS Docker script can also be used to create persistent sessions::

    fenicsproject create myproject
    fenicsproject run myproject

   To update the container to the newest version, run::

    fenicsproject pull quay.io/opentidalfarm/virtual

   To see all the options run::

    fenicsproject help

For more details and tips on how to work with FEniCS and Docker, see
our `FEniCS Docker page
<http://fenics-containers.readthedocs.org/en/latest/>`_.

Manual installation
-------------------

For the manual installation of OpenTidalFarm, we recommend to use Ubuntu Linux.
First, you need to install these OpenTidalFarm dependencies:

- `FEniCS`_ (Follow the Ubuntu PPA installation)
- `dolfin-adjoint/libadjoint`_ (Follow the Ubuntu PPA installation)
- `SciPy`_::

    pip install --user scipy

- `pyyaml`_::

     pip install --user pyyaml

- `Uptide`_::

     pip install --user git+git://github.com/stephankramer/uptide.git

- `UTM`_::

     pip install --user utm


Finally, you can install the most recent version of OpenTidalFarm with:

.. code-block:: bash

   pip install --user https://github.com/OpenTidalFarm/OpenTidalFarm/archive/master.zip

Once finished, you can test the installation with:

.. code-block:: bash

    python -c "import opentidalfarm"

If no errors occur, your installation was succesfull.

One can download the examples with all mesh data with these commands:

.. code-block:: bash

   git clone https://github.com/OpenTidalFarm/OpenTidalFarm.git
   cd OpenTidalFarm
   git submodule init
   git submodule update

The examples are then stored in the "OpenTidalFarm/examples" directory.

.. _Ubuntu: http://www.ubuntu.com/
.. _FEniCS: http://fenicsproject.org/download/
.. _dolfin-adjoint: http://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html
.. _Uptide: https://github.com/stephankramer/uptide
.. _UTM: https://pypi.python.org/pypi/utm
.. _Download OpenTidalFarm: https://github.com/funsim/OpenTidalFarm/zipball/master
.. _Issue tracker: https://github.com/OpenTidalFarm/OpenTidalFarm/issues
.. _SciPy: http://www.scipy.org
.. _pyyaml: http://pyyaml.org

Older versions
--------------

Version 1.5

.. code-block:: bash

   pip install --user https://github.com/OpenTidalFarm/OpenTidalFarm/archive/opentidalfarm-1.5.zip

Version 1.4

.. code-block:: bash

   pip install --user https://github.com/OpenTidalFarm/OpenTidalFarm/archive/opentidalfarm-1.4.zip

Version 0.9.1

.. code-block:: bash

   pip install --user https://github.com/OpenTidalFarm/OpenTidalFarm/archive/opentidalfarm-0.9.1.zip
