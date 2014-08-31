Installation
============

It is recommended to use OpenTidalFarm on the `Ubuntu`_ operating system.

Quick install
-------------

Install using git:

.. code-block:: bash

   git clone git@github.com:OpenTidalFarm/OpenTidalFarm.git
   cd OpenTidalFarm
   git submodule init
   git submodule update
   python setup.py install

Install using pip:

.. code-block:: bash

   pip install git+git://github.com/OpenTidalFarm/OpenTidalFarm.git

Test the installation with

.. code-block:: bash

    python -c "import opentidalfarm"

If no errors occur, your installation was succesfull.    

Dependencies
------------

OpenTidalFarm depends on following packages:

- `FEniCS`_ (Follow the Ubuntu PPA installation)
- `dolfin-adjoint`_ (Follow the Ubuntu PPA installation)
- `SciPy >=0.11`_ - e.g. with:

   ``sudo pip install scipy``

- `Uptide`_
- `UTM`_ - e.g. with:

   ``sudo pip install utm``

.. _Ubuntu: http://www.ubuntu.com/
.. _FEniCS: http://fenicsproject.org/download/
.. _dolfin-adjoint: http://dolfin-adjoint.org/download/index.html
.. _SciPy >=0.11: https://github.com/scipy/scipy
.. _Uptide: https://github.com/stephankramer/uptide
.. _UTM: https://pypi.python.org/pypi/utm
.. _Download OpenTidalFarm: https://github.com/funsim/OpenTidalFarm/zipball/master

