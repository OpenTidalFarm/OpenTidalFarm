Installation
============

Note: This installation procedure assumes that you are running the `Ubuntu`_ operating system.

The installation consists of following steps:

1. Download and install the dependencies:
    - `FEniCS`_ (Follow the Ubuntu PPA installation order to get a recent installation)
    - `dolfin-adjoint`_
    - `SciPy >=0.11`_ - e.g. with:

       ``sudo pip install scipy``

    - `Uptide`_
    - `UTM`_ - e.g. with:

       ``sudo pip install utm``

2. `Download OpenTidalFarm`_ and extract it.

3. Open a terminal and change into the extracted directory and run:

   ``sudo python setup.py install``

   to install it. A simple test to check if the installation was correct is to open a Python shell and type:

   ``from opentidalfarm import *``

   If you get an error, make sure that you have set the `PYTHONPATH` correctly. In Linux, this can be done with:

   ``export PYTHONPATH=/XYZ:$PYTHONPATH``

   where ``XYZ`` should be replaced with the path to your OpenTidalFarm installation. 

.. _Ubuntu: http://www.ubuntu.com/
.. _FEniCS project >=1.2: http://fenicsproject.org/download/
.. _dolfin-adjoint: http://dolfin-adjoint.org/download/index.html
.. _SciPy Version >=0.11: https://github.com/scipy/scipy
.. _Uptide: https://github.com/stephankramer/uptide
.. _UTM: https://pypi.python.org/pypi/utm
.. _Download OpenTidalFarm: https://github.com/funsim/OpenTidalFarm/zipball/master

