Installation
============

It is recommended to use OpenTidalFarm on the `Ubuntu`_ operating system.

Quick install
-------------

Manual installation (recommended)
*********************************

For the manual installation of OpenTidalFarm you need to have the Ubuntu Linux.
Then you need to install these OpenTidalFarm dependencies:

- `FEniCS`_ (Follow the Ubuntu PPA installation)
- `dolfin-adjoint`_ (Follow the Ubuntu PPA installation)
- `SciPy`_

   ``pip install --user scipy``

- `pyyaml`_

   ``pip install --user pyyaml``

- `Uptide`_

   ``pip install --user git+git://github.com/stephankramer/uptide.git``

- `UTM`_

   ``pip install --user utm``


Finally, you can install the most recent version of OpenTidalFarm with:

.. code-block:: bash

   pip install --user https://github.com/OpenTidalFarm/OpenTidalFarm/archive/master.zip

Once finished, you can test the installation with:

.. code-block:: bash

    python -c "import opentidalfarm"

If no errors occur, your installation was succesfull.

One can download the examples with all mesh data with these commands:

.. code-block:: bash

   git clone git@github.com:OpenTidalFarm/OpenTidalFarm.git
   cd OpenTidalFarm
   git submodule init
   git submodule update

The examples are then stored in the "OpenTidalFarm/examples" directory.

.. _Ubuntu: http://www.ubuntu.com/
.. _FEniCS: http://fenicsproject.org/download/
.. _dolfin-adjoint: http://www.dolfin-adjoint.org/download/index.html
.. _Uptide: https://github.com/stephankramer/uptide
.. _UTM: https://pypi.python.org/pypi/utm
.. _Download OpenTidalFarm: https://github.com/funsim/OpenTidalFarm/zipball/master
.. _Issue tracker: https://github.com/OpenTidalFarm/OpenTidalFarm/issues
.. _SciPy: http://www.scipy.org
.. _pyyaml: http://pyyaml.org

Automatic installation using hashdist (experimental)
****************************************************

Following command will install OpenTidalFarm and all its dependencies into a isolated environment on your computer.

.. code-block:: bash

   curl -s https://bitbucket.org/simon_funke/fenics-developer-tools/raw/master/opentidalfarm-install.sh | bash

Select the option "[1] latest stable version of FEniCS" during the installation.

Once finished, you can test the installation with:

.. code-block:: bash

    python -c "import opentidalfarm"

If no errors occur, your installation was succesfull.

If you have any problems with the installation, please use our `Issue tracker`_.

Older versions
**************

Version 1.5

.. code-block:: bash

   pip install --user https://github.com/OpenTidalFarm/OpenTidalFarm/archive/opentidalfarm-1.5.zip

Version 1.4

.. code-block:: bash

   pip install --user https://github.com/OpenTidalFarm/OpenTidalFarm/archive/opentidalfarm-1.4.zip

Version 0.9.1

.. code-block:: bash

   pip install --user https://github.com/OpenTidalFarm/OpenTidalFarm/archive/opentidalfarm-0.9.1.zip
