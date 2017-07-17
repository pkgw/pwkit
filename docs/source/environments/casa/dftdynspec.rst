.. Copyright 2017 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

CASA: DFT Dynamic Spectra (:mod:`pwkit.environments.casa.dftdynspec`)
================================================================================

.. automodule:: pwkit.environments.casa.dftdynspec
   :synopsis: dynamic spectra of point sources via DFT

.. currentmodule:: pwkit.environments.casa.dftdynspec


The function :func:`dftdynspec` computes the dynamic spectrum of a point
source using a discrete Fourier transform of its visibilities. The function
:func:`dftdynspec_cli` provides a hook to launch the computation from a
command-line program.

You can launch a computation from the command line using the command
``casatask dftdynspec``.

Due to limitations in the documentation system we’re using, the options to the
dynamic spectrum computation are not documented here. You can read about them
by running ``casatask dftdynspec --help``.


The :class:`Loader` class
-------------------------

Unlike the other DFT tasks, :func:`dftdynspec` produces output that is not
easily represented as a table. It is saved to disk as a set of Numpy arrays.
The :class:`Loader` class provides a convenient mechanism for loading an
output data set.

To load and manipulate data, create a :class:`Loader` instance and then access
the various arrays described below::

  from pwkit.environments.casa.dftdynspec import Loader
  path = 'mydataset.npy' # this gets customized
  ds = Loader(path)
  print('Maximum real part:', ds.reals.max())

.. autoclass:: Loader
   :members:

   **Members**
