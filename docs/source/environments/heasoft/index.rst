.. Copyright 2015-2017 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

HEASoft (:mod:`pwkit.environments.heasoft`)
========================================================================

.. automodule:: pwkit.environments.heasoft
   :synopsis: access to HEASoft via the :mod:`pwkit.environments` framework

.. currentmodule:: pwkit.environments.heasoft


Using HEASoft in the :mod:`pwkit.environments` framework
--------------------------------------------------------

The module :mod:`pwkit.environments` implements a system for running
sub-programs that depend on large, external software environments such as
HEASoft. It provides a command-line tool, ``pkenvtool``, that you can use to
run HEASoft code in a controlled environment.

In order to use this module, you must tell the :mod:`pwkit.environments`
system where your HEASoft installation may be found. To do this, just export
an environment variable named ``$PWKIT_HEASOFT`` that stores the path to the
*platform-specific* subdirectory of your HEASoft installation. In other words,
the file ``$PWKIT_HEASOFT/headas-init.sh`` should exist. On a Linux system the
value of ``$PWKIT_HEASOFT`` might end with something like
``x86_64-unknown-linux-gnu-libc2.23``. Once you’ve correctly set this
environment variable, the environments system will take care of the rest.


HEAsoft installation notes
--------------------------

The following examples assume version 6.21 for concreteness. Substitute your
actual version as needed, of course.

Installation of HEASoft from source is strongly recommended. Download the
source code from a URL like `this one
<https://heasarc.gsfc.nasa.gov/FTP/software/lheasoft/lheasoft6.31/heasoft-6.31.1src.tar.gz>`_.
The HEASoft website lets you customize the tarball, but it’s probably easiest
just to do the full install every time. The tarball unpacks into a directory
named like ``heasoft-6.21/...`` so you can safely ``curl|tar`` in your
source-code directory.

To build, then run something like::

  $ cd heasoft-6.21/BUILD_DIR
  $ ./configure --prefix=/a/heasoft/6.21
  $ make # note: not parallel-friendly
  $ make install

You then need to fetch the ``CALDB`` data files into the HEASoft installation directory::

  $ cd /a/heasoft/6.21
  $ wget http://heasarc.gsfc.nasa.gov/FTP/caldb/software/tools/caldb.config
  $ wget http://heasarc.gsfc.nasa.gov/FTP/caldb/software/tools/alias_config.fits
