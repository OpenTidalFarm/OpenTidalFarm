Contributing to OpenTidalFarm
=============================

If you wish to contribute to the development of OpenTidalFarm please adhere to
the following *guidelines* on Python *coding style* and *language rules*.

It is **strongly** recommended that contributors read through the entirety of
the `Google Python Style Guide
<http://google-styleguide.googlecode.com/svn/trunk/pyguide.html>`_.

Key points are summarised below.


Style Guide
-----------

Formatting
^^^^^^^^^^

**Line length** should be limited to **80** characters. Use Python's implicit
line joining inside parentheses, brackets and braces.

.. code-block:: python

  string = ("This is a very long string containing an example of how to "
            "implicitly join strings over multiple lines using parentheses.")

Similarly for a long if statement:

.. code-block:: python

  if (we_have_long_variable_names or lots_of_comparisons and
      need_to_break_onto_another_line):
      # We should use Python's implicit line joining within parentheses.

It is permissible to use more than 80 characters in a line when including a
URL in a comment, for example:

.. code-block:: python

  # Find more information at:
  # http://www.a-ridiculously-long-url-which-spans-more-than-80-characters-is-allowed-in-a-comment.com


**Indentation** should be with **4 spaces**. When a line is implicitly
continued (as demonstrated in the line length section) the wrapped elements
should be aligned vertically or using a hanging indent of 4 spaces.

.. code-block:: python

  # Aligned with the opening delimiter
  foo = long_function_name(variable_one, variable_two, variable_three,
                           variable_four, variable_five)

  # Aligned using a hanging indent of 4 spaces; nothing on the first line
  foo = long_function_name(
      variable_one, variable_two, variable_three, variable_four, variable_five)


**Whitespace** should follow normal typographic rules, i.e. put a space after
a comma but not before.

Do not put whitespace:

* inside parentheses, brackets, or braces,
* before a comma, colon or semicolon, and
* before opening parentheses that starts an argument list, indexing or slicing.

A single space should be added around binary operators for:

* assignment (``=``),
* comparisons (``==, <, >, !=, <>, <=, >=, in, not in, is, is not``), and
* Booleans (``and, or, not``).

However, spaces should *not* be added around the assignment operator (``=``)
when used to indicate a keyword argument or a default value. I.e. you should
do this:

.. code-block:: python

    functions_with_default_arguments(argument_one=10.0, argument_two=20.0)

Many more examples regarding whitespace may be again found in the
`whitespace
<http://google-styleguide.googlecode.com/svn/trunk/pyguide.html?showone=Whitespace#Whitespace>`_
section of the Google Python Style Guide.

**Blank lines** should be added as such:

* Two blank lines between top-level definitions, be they function or class
  definitions.
* One blank line between method definitions and between the class line and the
  first method. Use single blank lines as you judge appropriate within
  functions or methods.


Naming Convention
^^^^^^^^^^^^^^^^^

The following convention should be used for naming:

``module_name``, ``package_name``, ``ClassName``, ``method_name``,
``ExceptionName``, ``function_name``, ``GLOBAL_CONSTANT_NAME``,
``global_variable_name``, ``instance_variable_name``,
``function_parameter_name``, ``local_variable_name``.


Imports formatting
^^^^^^^^^^^^^^^^^^

Imports should be at the top of the file and should occur on separate lines:

.. code-block:: python

  import numpy
  import dolfin

They should also be ordered from most generic to least generic:

* standard library imports (such as ``math``),
* third-party imports (such as ``opentidalfarm``),
* application-specific imports (such as ``farm``).


Commenting and Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Documenting your work is crucial for to allowing other users and developers to
quickly understand what your work does and how it works. For example a
docstring for a function should give enough information to write a call it
without reading the function's code. A docstring should describe the
function's calling syntax and its semantics, not its implementation. For
tricky code, comments alongside the code are more appropriate than using
docstrings.

OpenTidalFarm uses Sphinx documentation thus a certain syntax is required,
examples are given below.

For a module:

.. code-block:: python

   """
   .. module:: example_module
      :synopsis: Brief description of the module.

   """

For a class:

.. code-block:: python

    class ExampleClass(object):
        """A brief description of the class.

        A longer description of the class.

        .. note::

            Any notes you may wish to highlight in the online documentation.

        """
        # Implementation of ExampleClass...

And an example for a function:

.. code-block:: python

    def public_function_with_sphinx_docstring(name, state=None):
        """This function does something.

        :param name: The name to use.
        :type name: str.
        :param state: Current state to be in.
        :type state: bool.
        :returns:  int -- the return code.
        :raises: AttributeError, KeyError

        """
        # Implementation of public_function_with_sphinx_docstring...


Finally, comments should also be added within the code to explain where it may
not be immediately obvious what is being done. These comments should be well
written with correct spelling, punctuation and grammar.


Language Rules
--------------

Most of the information regarding language rules in the `Google Python Style
Guide`_ is fairly obvious but a few important points are highlighted here.

**List comprehensions** when used correctly can create lists in a very concise
manner, however they should not be used in complicated situations as they can
become hard to read.

**Properties** may be used to **control access** to class data members. For
example a class which defines the turbine farm may be initialized with the
coordinates defining the boundary for the site. Once initialized it does not
make sense to resize the site (as turbines may no longer lie within its
bounds) but the user may wish to still access these values. In Python there is
no way to truly make certain data private but the following convention is
ususally adopted.

For read-only data the `property` decorator is used:

.. code-block:: python

  class Circle(object):
      def __init__(self, radius):
          self._radius = radius

      @property
      def radius(self):
          """The radius of the circle."""
          return self._radius

Thus the user may still access the radius of the circle without changing it:

.. code-block:: python

  >>> circle = Circle(10.0)
  >>> circle.radius
  10.0
  >>> circle.radius = 15.0
  AttributeError: can't set attribute


If the user wishes the provide full access to a data member it can be done so
using the built-in property function. This also provides a convenient way to
allow a number of properties to be based upon a single property.

.. code-block:: python

  class Circle(object):
      def __init__(self, radius):
          self._radius = radius

      def _get_radius(self):
          return self._radius

      def _set_radius(self, radius):
          self._radius = radius

      radius = property(_get_radius, _set_radius, "Radius of circle")

      def _get_diameter(self):
          return self._radius*2

      def _set_diameter(self, diameter):
          self._radius = diameter*0.5

      diameter = property(_get_diameter, _set_diameter, "Diameter of circle")

Thus we may do the following:

.. code-block:: python

  >>> circle = Circle(10.0)
  >>> circle.diameter
  20.0
  >>> circle.diameter = 10.0
  >>> circle.radius
  5.0


Logging using dolfin.log
------------------------

It is strongly encouraged that developers make use of the logging capability
of ``dolfin``. The verbosity of the logger during runtime may be altered by
the user allowing for easier debugging.

The logger is included by ``dolfin`` and has a number of verbosity levels
given in the table below.

=========== =====
 Log Level  Value
=========== =====
ERROR         40
WARNING       30
INFO          20
PROGRESS      16
DBG / DEBUG   10
=========== =====

Controlling the verbosity of what the logger displays during runtime is simple:

.. code-block:: python

  import dolfin
  # Can be any of the values from the table above
  dolfin.set_log_level(INFO)

Using the logger is simple, for example when adding turbines to a farm it may
be useful to know how many turbines are being added to the farm (for which we
would set the log level to INFO). In certain cases it may useful to know when
each turbine is being added, in which case we would use the PROGRESS log
level:

.. code-block:: python

  import dolfin

  class RectangularFarm(object):
      # Implementation of RectangularFarm ...


      def add_regular_turbine_layout(self, num_x, num_y):
          """Adds turbines to the farm in a regularly spaced array."""

          dolfin.log(dolfin.INFO, "Adding %i turbines to the farm..."
                     % (num_x*num_y))

          added = 1
          total = num_x*num_y
          for x in num_x:
              for y in num_y:
                  dolfin.log(dolfin.PROGRESS, "Adding turbine %i of %i..."
                             % (added, total))
                  # ...add turbines to the farm
                  added += 1

          dolfin.log(dolfin.INFO, "Added %i turbines to the farm."
                     % (added))


More information may be found in the documentation.

It is also suggested that for computationally expensive functions that the
``dolfin.Progress`` bar is used. An example from the `documentation
<http://fenicsproject.org/documentation/dolfin/1.0.1/python/programmers-reference/cpp/Progress.html>`_
is shown below.

.. code-block:: python

  >>> import dolfin
  >>> dolfin.set_log_level(dolfin.PROGRESS)
  >>> n = 10000000
  >>> progress_bar = dolfin.Progress("Informative progress message...", n)
  >>> for i in range(n):
  ...     progress_bar += 1
  ...
  Informative progress message... [>                                    ] 0.0%
  Informative progress message... [=>                                   ] 5.2%
  Informative progress message... [====>                                ] 11.1%
  Informative progress message... [======>                              ] 17.0%

Adding documented examples
--------------------------

The documentation for examples is automatically generated from the source code
using `pylit <https://pypi.python.org/pypi/pylit>`_. 

Follow these steps to add an example:

1. Create a new subdirectory in ``examples/`` and add the documented Python source
   code (use for example existing examples for references).
2. Add the example to the `build_examples` task in ``docs/Makefile`` (again use 
   existing commands as a template).
3. Add the example into the list in ``examples.rst`` to add the hyperlink.
4. Run "make html" in ``docs/``, check that the documentation looks as expected
   (open ``_build/html/index.html`` in an webbrowser).
5. Add the generated rst file in ``docs/examples/.../`` to the git repository.
   Commit, and check that the documentation is correct in the readthedocs
   OpenTidalFarm documentation.
