Contributing to OpenTidalFarm
=============================

If you wish to contribute to the development of OpenTidalFarm please adhere to
the following *guidelines* on Python *coding style* and *language rules*.

It is **strongly** recommended that contributors read through the entirety of
the `Google Python Style Guide`_.

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
`whitespace`_ section of the `Google Python Style Guide`_.

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
   ``.. module::`` example_module
   ``:synopsis:`` Brief description of the module.

   """

For a class:

.. code-block:: python

    class ExampleClass(object):
        """A brief description of the class.

        A longer description of the class.

        ``.. note::``

            Any notes you may wish to highlight in the online documentation.

        """
        # Implementation of ExampleClass...

And an example for a function:

.. code-block:: python

    def public_function_with_sphinx_docstring(name, state=None):
        """This function does something.

        ``:param name:`` The name to use.
        ``:type name:`` str.
        ``:param state:`` Current state to be in.
        ``:type state:`` bool.
        ``:returns:``  int -- the return code.
        ``:raises:`` AttributeError, KeyError

        """
        # Implementation of public_function_with_sphinx_docstring...


Finally, comments should also be added within the code to explain where it may
not be immediately obvious what is being done. These comments should be well
written with correct spelling, punctuation and grammar.



.. _Google Python Style Guide: http://google-styleguide.googlecode.com/svn/trunk/pyguide.html
.. _whitespace: http://google-styleguide.googlecode.com/svn/trunk/pyguide.html?showone=Whitespace#Whitespace
