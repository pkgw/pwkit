.. Copyright 2017 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

CASA
==============================================================================

The :mod:`pwkit.environments.casa` package provides convenient interfaces to
the `CASA`_ package for analysis of radio interferometric data. In particular,
it makes it much easier to build scripts and modules for automated data
analysis.

.. _CASA: https://casa.nrao.edu/

This module does *not* require a full CASA installation, but it does depend on
the availability of the ``casac`` Python module, which provides Python access
to the C++ code that drives most of CASA’s low-level functionality. By far the
easiest way to obtain this module is to use an installation of `Anaconda or
Miniconda Python`_ and install the `casa-python`_ package provided by Peter
Williams, which builds on the infrastructure provided by the `conda-forge`_
project.

.. _Anaconda or Miniconda Python: http://conda.pydata.org/miniconda.html
.. _casa-python: https://anaconda.org/pkgw-forge/casa-python
.. _conda-forge: https://conda-forge.github.io/

Alternatively, you can try to install CASA and extract the ``casac`` module
from its files `as described here`_. Or you can try to install *this module*
inside the Python environment bundled with CASA. Or you can compile and
underlying CASA C++ code yourself. But, using the pre-built packages is going
to be by far the simplest approach and is **strongly** recommended.

.. _as described here: https://newton.cx/~peter/2014/02/casa-in-python-without-casapy/


Outline of functionality
------------------------

This package provides several kinds of functionality.

- The :mod:`pwkit.environments.casa.tasks` module provides straightforward
  programmatic access to a wide selection of commonly-used CASA takes like
  ``gaincal`` and ``setjy``.
- ``pwkit`` installs a command-line program, ``casatask``, which provides
  command-line access to the tasks implemented in the
  :mod:`~pwkit.environments.casa.tasks` module, much as MIRIAD tasks can be
  driven straight from the command line.
- The :mod:`pwkit.environments.casa.util` module provides the lowest-level access
  to the “tool” structures defined in the C++ code.
- Several modules like :mod:`pwkit.environments.casa.dftphotom` provide
  original analysis features; :mod:`~pwkit.environments.casa.dftphotom`
  extracts light curves of point sources from calibrated visibility data.
- If you do have a full CASA installation available on your compuer, the
  :mod:`pwkit.environments.casa.scripting` module allows you to drive it from
  Python code in a way that allows you to analyze its output, check for
  error conditions, and so on. This is useful for certain features that are not
  currently available in the :mod:`~pwkit.environments.casa.tasks` module.


.. toctree::
   :maxdepth: 2

   ../pwkit/environments/casa/toplevel.rst
   ../pwkit/environments/casa/dftphotom.rst
   ../pwkit/environments/casa/scripting.rst
   ../pwkit/environments/casa/spwglue.rst
   ../pwkit/environments/casa/tasks.rst
   ../pwkit/environments/casa/util.rst
