Installation
============

It is recommended to use OpenTidalFarm on the `Ubuntu`_ operating system.

Quick install
-------------

Installation from source
************************

Following command will install OpenTidalFarm and all its dependencies into a isolated environment on your computer.

.. code-block:: bash

   curl -s https://bitbucket.org/simon_funke/fenics-developer-tools/raw/master/opentidalfarm-install.sh | bash



Install using pip
*****************

You can install OpenTidalFarm via pip or git, however make sure that you
also install the :ref:`dependencies`:

Using pip:

.. code-block:: bash

   pip install git+git://github.com/OpenTidalFarm/OpenTidalFarm.git

Using git:

.. code-block:: bash

   git clone git@github.com:OpenTidalFarm/OpenTidalFarm.git
   cd OpenTidalFarm
   git submodule init
   git submodule update
   python setup.py install


Test installation
-----------------

.. code-block:: bash

    python -c "import opentidalfarm"

If no errors occur, your installation was succesfull.

.. _dependencies:

Dependencies
------------

OpenTidalFarm depends on following packages:

- `FEniCS`_ (Follow the Ubuntu PPA installation)
- `dolfin-adjoint`_ (Follow the Ubuntu PPA installation)
- `SciPy >=0.11`_ - e.g. with:

   ``pip install scipy``

- `Uptide`_

   ``pip install git+git://github.com/stephankramer/uptide.git``

- `UTM`_ - e.g. with:

   ``pip install utm``

.. _Ubuntu: http://www.ubuntu.com/
.. _FEniCS: http://fenicsproject.org/download/
.. _dolfin-adjoint: http://dolfin-adjoint.org/download/index.html
.. _SciPy >=0.11: https://github.com/scipy/scipy
.. _Uptide: https://github.com/stephankramer/uptide
.. _UTM: https://pypi.python.org/pypi/utm
.. _Download OpenTidalFarm: https://github.com/funsim/OpenTidalFarm/zipball/master
