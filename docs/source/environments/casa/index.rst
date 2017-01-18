.. Copyright 2017 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

CASA (:mod:`pwkit.environments.casa`)
==============================================================================

.. automodule:: pwkit.environments.casa
   :synopsis: access to CASA via the :mod:`pwkit.environments` framework

.. currentmodule:: pwkit.environments.casa

.. toctree::
   :maxdepth: 2

   tasks.rst
   ../../pwkit/environments/casa/dftphotom.rst
   ../../pwkit/environments/casa/scripting.rst
   ../../pwkit/environments/casa/spwglue.rst
   ../../pwkit/environments/casa/util.rst


Using CASA in the :mod:`pwkit.environments` framework
-----------------------------------------------------

To use, export an environment variable ``$PWKIT_CASA`` pointing to the CASA
installation root. The files ``$PWKIT_CASA/asdm2MS`` and
``$PWKIT_CASA/casapy`` should exist.

**Note**: does this work on 32-bit systems? Does this work on Macs?


CASA installation notes
-----------------------

Download tarball as linked `from here
<http://casa.nrao.edu/casa_obtaining.shtml>`_. The tarball unpacks to some
versioned subdirectory. The names and version codes are highly variable and
annoying.
