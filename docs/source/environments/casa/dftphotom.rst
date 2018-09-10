.. Copyright 2015-2018 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Compact-source photometry with discrete Fourier transformations (``pwkit.environments.casa.dftphotom``)
=======================================================================================================

.. automodule:: pwkit.environments.casa.dftphotom
   :synopsis: Light curves from interferometric visibilities

.. currentmodule:: pwkit.environments.casa.dftphotom


Usage from Within Python
------------------------

Basic usage from within Python looks like this::

  from pwkit.astutil import parsehours, parsedeglat
  from pwkit.environments.casa import dftphotom

  # Here's a sample way to specify the coordinates to
  # use; anything that produces J2000 RA/Dec in radians
  # will work:
  ra = parsehours('17:45:00') # result is in radians
  dec = parsedeglat('-23:00:00') # result is in radians

  cfg = dftphotom.Config()
  cfg.vis = 'path/to/vis/data.ms'
  cfg.format = dftphotom.PandasOutputFormat()
  cfg.outstream = open('mydata.txt', 'w')
  cfg.rephase = (ra, dec)

  dftphotom.dftphotom(cfg)

The main algorithm is implemented in the :func:`dftphotom` function. All of
the algorithm parameters are passed to the function via a :class:`Config`
structure. You can create one :class:`Config` and call :func:`dftphotom` with
it repeatedly, altering the parameters each time if you have a series of
related computations to run.


API Reference
-------------

.. autoclass:: Config

   .. autoattribute:: vis
   .. autoattribute:: datacol
   .. autoattribute:: believeweights

   .. attribute:: outstream

      A file-like object into which the output table of data will be written.
      No default in the Python interface.

   .. autoattribute:: datascale

   .. attribute:: format

      An instance of a class that will format the algorithm outputs into text.
      Either :class:`HumaneOutputFormat` or :class:`PandasOutputFormat`.

   .. attribute:: rephase

      A coordinate tuple ``(ra, dec)``, giving a location towards which to
      rephase the visibility data. The inputs are in radians. If left as
      ``None``, the visibilities will not be rephased.

   .. rubric:: Generic CASA data-selection options

   .. autoattribute:: array
   .. autoattribute:: baseline
   .. autoattribute:: field
   .. autoattribute:: observation
   .. autoattribute:: polarization
   .. autoattribute:: scan
   .. autoattribute:: scanintent
   .. autoattribute:: spw
   .. autoattribute:: taql
   .. autoattribute:: time
   .. autoattribute:: uvdist

   .. rubric:: Generic CASA task options

   .. autoattribute:: loglevel


.. autofunction:: dftphotom

.. autofunction:: dftphotom_cli

.. autoclass:: HumaneOutputFormat

.. autoclass:: PandasOutputFormat
