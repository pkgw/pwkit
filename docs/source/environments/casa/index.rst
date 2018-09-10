.. Copyright 2017 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

CASA (:mod:`pwkit.environments.casa`)
==============================================================================

.. automodule:: pwkit.environments.casa
   :synopsis: access to CASA via the :mod:`pwkit.environments` framework

.. currentmodule:: pwkit.environments.casa


More detailed documentation
---------------------------

.. toctree::
   :maxdepth: 2

   tasks.rst
   utilities.rst
   dftdynspec.rst
   dftphotom.rst
   ../../pwkit/environments/casa/scripting.rst
   ../../pwkit/environments/casa/spwglue.rst


Using CASA in the :mod:`pwkit.environments` framework
-----------------------------------------------------

The module :mod:`pwkit.environments` implements a system for running
sub-programs that depend on large, external software environments such as
CASA. It provides a command-line tool, ``pkenvtool``, that you can use to run
code in a controlled CASA environment.

Some of the :doc:`tasks <tasks>` provided by :mod:`pwkit` rely on this
framework to implement their functionality — in these cases, the value that
:mod:`pwkit` is providing is that it lets you access complex CASA
functionality through a simple function call in a standard Python environment,
rather than requiring manual invocation in a ``casapy`` shell.

In order to use these tasks or the CASA features of the ``pkenvtool`` program,
you must tell the :mod:`pwkit.environments` system where your CASA
installation may be found. To do this, just export an environment variable
named ``$PWKIT_CASA`` that stores the path to the CASA installation root. In
other words, the file ``$PWKIT_CASA/bin/casa`` should exist. (Well, the code
also checks for ``$PWKIT_CASA/bin/casapy`` to try to be compatible with older
CASA versions.) The environments system will take care of the rest.

**Note**: does this work on 32-bit systems? Does this work on Macs?


CASA installation notes
-----------------------

Download tarball as linked `from here
<http://casa.nrao.edu/casa_obtaining.shtml>`_. The tarball unpacks to some
versioned subdirectory. The names and version codes are highly variable and
annoying.
