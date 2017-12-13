.. Copyright 2015-2017 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

.. _casa-tasks:

Programmatic access to CASA tasks (:mod:`pwkit.environments.casa.tasks`)
========================================================================

.. automodule:: pwkit.environments.casa.tasks
   :synopsis: programmatic access to CASA tasks

.. currentmodule:: pwkit.environments.casa.tasks

Example programmatic usage::

  from pwkit.environments.casa import tasks

  vis_path = 'mydataset.ms'

  # A basic listobs:

  for output_line in tasks.listobs(vis_path):
      print(output_line)

  # Split a dataset with filtering and averaging:

  cfg = tasks.SplitConfig()
  cfg.vis = vis_path
  cfg.out = 'new-' + vis_path
  cfg.spw = '0~8'
  cfg.timebin = 60 # seconds
  tasks.split(cfg)

This module implements the following analysis tasks. Some of them are
extremely close to CASA tasks of the same name; some are streamlined; some are
not provided in CASA at all.

- :ref:`task-applycal` — use calibration tables to generate ``CORRECTED_DATA``
  from ``DATA``.
- :ref:`task-bpplot` — plot a bandpass calibration table; an order of
  magnitude faster than the CASA equivalent.
- :ref:`task-clearcal` — fill calibration tables with default.
- :ref:`task-concat` — concatenate two data sets.
- :ref:`task-delcal` — delete the ``MODEL_DATA`` and/or ``CORRECTED_DATA`` MS
  columns.
- :ref:`task-elplot` — plot elevations of the fields observed in an MS.
- :ref:`task-extractbpflags` — extract a table of channel flags from a
  bandpass calibration table.
- :ref:`task-flagcmd` — apply flags to an MS using a generic infrastructure.
- :ref:`task-flaglist` — apply a textual list of flag commands to an MS.
- :ref:`task-flagzeros` — flag zero-valued visibilites in an MS.
- :ref:`task-fluxscale` — use a flux density model to absolutely scale a gain
  calibration table.
- :ref:`task-ft` — generate model visibilities from an image.
- :ref:`task-gaincal` — solve for a gain calibration table.
- :ref:`task-gencal` — generate various calibration tables that do not depend on
  the actual visibility data in an MS.
- :ref:`task-getopacities` — estimate atmospheric opacities for an
  observation.
- :ref:`task-gpdetrend` — remove long-term phase trends from a complex-gain
  calibration table.
- :ref:`task-gpplot` — plot a complex-gain calibration table in a sensible
  way.
- :ref:`task-image2fits` — convert a CASA image to FITS format.
- :ref:`task-importalma` — convert an ALMA SDM file to MS format.
- :ref:`task-importevla` — convert an EVLA SDM file to MS format.
- :ref:`task-listobs` — print out the basic observational characteristics in
  an MS data set.
- :ref:`task-listsdm` — print out the basic observational characteristics in
  an SDM data set.
- :ref:`task-mfsclean` — image calibrated data using MFS and CLEAN.
- :ref:`task-mjd2date` — convert an MJD to a date in the textual format used by CASA.
- :ref:`task-mstransform` — perform basic streaming transforms on an MS data,
  such as time averaging, Hanning smoothing, and/or velocity resampling.
- :ref:`task-plotants` — plot the positions of the antennas used in an MS.
- :ref:`task-plotcal` — plot a complex-gain calibration table using CASA’s
  default infrastructure.
- :ref:`task-setjy` — insert absolute flux density calibration information
  into a dataset.
- :ref:`task-split` — extract a subset of an MS.
- :ref:`task-tsysplot` — plot how the typical system temperature varies over
  time.
- :ref:`task-uvsub` — fill ``CORRECTED_DATA`` with ``DATA - MODEL_DATA``.
- :ref:`task-xyphplot` — plot a frequency-dependent X/Y phase calibration
  table.

The following tasks are provided by the associated command line program,
``casatask``, but do not have dedicated functions in this module.

- ``closures`` — see :mod:`~pwkit.environment.casa.closures`.
- :ref:`task-delmod` — this is too trivial to need its own function.
- ``dftdynspec`` — see :mod:`~pwkit.environment.casa.dftdynspec`.
- ``dftphotom`` — see :mod:`~pwkit.environment.casa.dftphotom`.
- ``dftspect`` — see :mod:`~pwkit.environment.casa.dftspect`.
- :ref:`task-flagmanager` — more specialized functions should be used in code.
- ``gpdiagnostics`` — see :mod:`~pwkit.environment.casa.gpdiagnostics`.
- ``polmodel`` — see :mod:`~pwkit.environment.casa.polmodel`.
- ``spwglue`` — see :mod:`~pwkit.environment.casa.spwglue`.


Tasks
-----

This documentation is automatically generated from text that is targeted at
the command-line tasks, and so may read a bit strangely at times.

.. _task-applycal:

applycal
~~~~~~~~~~~~~~~~~~~
.. autofunction:: applycal
.. autoclass:: ApplycalConfig


.. _task-bpplot:

bpplot
~~~~~~~~~~~~~~~~~~~
.. autofunction:: bpplot
.. autoclass:: BpplotConfig

.. _task-clearcal:

clearcal
~~~~~~~~~~~~~~~~~~~
.. autofunction:: clearcal

.. _task-concat:

concat
~~~~~~~~~~~~~~~~~~~
.. autofunction:: concat

.. _task-delcal:

delcal
~~~~~~~~~~~~~~~~~~~
.. autofunction:: delcal

.. _task-delmod:

delmod
~~~~~~~~~~~~~~~~~~~
.. autofunction:: delmod_cli

.. _task-elplot:

elplot
~~~~~~~~~~~~~~~~~~~
.. autofunction:: elplot
.. autoclass:: ElplotConfig

.. _task-extractbpflags:

extractbpflags
~~~~~~~~~~~~~~~~~~~
.. autofunction:: extractbpflags

.. _task-flagcmd:

flagcmd
~~~~~~~~~~~~~~~~~~~
.. autofunction:: flagcmd
.. autoclass:: FlagcmdConfig

.. _task-flaglist:

flaglist
~~~~~~~~~~~~~~~~~~~
.. autofunction:: flaglist
.. autoclass:: FlaglistConfig

.. _task-flagmanager:

flagmanager
~~~~~~~~~~~~~~~~~~~
.. autofunction:: flagmanager_cli

.. _task-flagzeros:

flagzeros
~~~~~~~~~~~~~~~~~~~
.. autofunction:: flagzeros
.. autoclass:: FlagzerosConfig

.. _task-fluxscale:

fluxscale
~~~~~~~~~~~~~~~~~~~
.. autofunction:: fluxscale
.. autoclass:: FluxscaleConfig

.. _task-ft:

ft
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ft
.. autoclass:: FtConfig

.. _task-gaincal:

gaincal
~~~~~~~~~~~~~~~~~~~
.. autofunction:: gaincal
.. autoclass:: GaincalConfig

.. _task-gencal:

gencal
~~~~~~~~~~~~~~~~~~~
.. autofunction:: gencal
.. autoclass:: GencalConfig

.. _task-getopacities:

getopacities
~~~~~~~~~~~~~~~~~~~
.. autofunction:: getopacities

.. _task-gpdetrend:

gpdetrend
~~~~~~~~~~~~~~~~~~~
.. autofunction:: gpdetrend
.. autoclass:: GpdetrendConfig

.. _task-gpplot:

gpplot
~~~~~~~~~~~~~~~~~~~
.. autofunction:: gpplot
.. autoclass:: GpplotConfig

.. _task-image2fits:

image2fits
~~~~~~~~~~~~~~~~~~~
.. autofunction:: image2fits

.. _task-importalma:

importalma
~~~~~~~~~~~~~~~~~~~
.. autofunction:: importalma

.. _task-importevla:

importevla
~~~~~~~~~~~~~~~~~~~
.. autofunction:: importevla

.. _task-listobs:

listobs
~~~~~~~~~~~~~~~~~~~
.. autofunction:: listobs

.. _task-listsdm:

listsdm
~~~~~~~~~~~~~~~~~~~
.. autofunction:: listsdm

.. _task-mfsclean:

mfsclean
~~~~~~~~~~~~~~~~~~~
.. autofunction:: mfsclean
.. autoclass:: MfscleanConfig

.. _task-mjd2date:

mjd2date
~~~~~~~~~~~~~~~~~~~
.. autofunction:: mjd2date

.. _task-mstransform:

mstransform
~~~~~~~~~~~~~~~~~~~
.. autofunction:: mstransform
.. autoclass:: MstransformConfig

.. _task-plotants:

plotants
~~~~~~~~~~~~~~~~~~~
.. autofunction:: plotants

.. _task-plotcal:

plotcal
~~~~~~~~~~~~~~~~~~~
.. autofunction:: plotcal
.. autoclass:: PlotcalConfig

.. _task-setjy:

setjy
~~~~~~~~~~~~~~~~~~~
.. autofunction:: setjy
.. autoclass:: SetjyConfig

.. _task-split:

split
~~~~~~~~~~~~~~~~~~~
.. autofunction:: split
.. autoclass:: SplitConfig

.. _task-tsysplot:

tsysplot
~~~~~~~~~~~~~~~~~~~
.. autofunction:: tsysplot
.. autoclass:: TsysplotConfig

.. _task-uvsub:

uvsub
~~~~~~~~~~~~~~~~~~~
.. autofunction:: uvsub
.. autoclass:: UvsubConfig

.. _task-xyphplot:

xyphplot
~~~~~~~~~~~~~~~~~~~
.. autofunction:: xyphplot
.. autoclass:: XyphplotConfig
