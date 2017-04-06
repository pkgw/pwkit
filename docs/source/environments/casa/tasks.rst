.. Copyright 2015-2017 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

.. _casa-tasks:

Programmatic access to CASA tasks (:mod:`pwkit.environments.casa.tasks`)
========================================================================

.. automodule:: pwkit.environments.casa.tasks
   :synopsis: programmatic access to CASA tasks

.. currentmodule:: pwkit.environments.casa.tasks

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

The documentation scheme for the CASA task wrappers is a work in progress. For
now, the best way to get the best documentation is from the command-line
interface. Run ``casatask taskname --help``, replacing ``taskname`` with the
name of the task of interest.

.. _task-applycal:

applycal
~~~~~~~~~~~~~~~~~~~

.. _task-bpplot:

bpplot
~~~~~~~~~~~~~~~~~~~

.. _task-clearcal:

clearcal
~~~~~~~~~~~~~~~~~~~

.. _task-concat:

concat
~~~~~~~~~~~~~~~~~~~

.. _task-delcal:

delcal
~~~~~~~~~~~~~~~~~~~

.. _task-delmod:

delmod
~~~~~~~~~~~~~~~~~~~

.. _task-elplot:

elplot
~~~~~~~~~~~~~~~~~~~

.. _task-extractbpflags:

extractbpflags
~~~~~~~~~~~~~~~~~~~

.. _task-flagcmd:

flagcmd
~~~~~~~~~~~~~~~~~~~

.. _task-flaglist:

flaglist
~~~~~~~~~~~~~~~~~~~

.. _task-flagmanager:

flagmanager
~~~~~~~~~~~~~~~~~~~

.. _task-flagzeros:

flagzeros
~~~~~~~~~~~~~~~~~~~

.. _task-fluxscale:

fluxscale
~~~~~~~~~~~~~~~~~~~

.. _task-ft:

ft
~~~~~~~~~~~~~~~~~~~

.. _task-gaincal:

gaincal
~~~~~~~~~~~~~~~~~~~

.. _task-gencal:

gencal
~~~~~~~~~~~~~~~~~~~

.. _task-getopacities:

getopacities
~~~~~~~~~~~~~~~~~~~

.. _task-gpdetrend:

gpdetrend
~~~~~~~~~~~~~~~~~~~

.. _task-gpplot:

gpplot
~~~~~~~~~~~~~~~~~~~

.. _task-image2fits:

image2fits
~~~~~~~~~~~~~~~~~~~

.. _task-importalma:

importalma
~~~~~~~~~~~~~~~~~~~

.. _task-importevla:

importevla
~~~~~~~~~~~~~~~~~~~

.. _task-listobs:

listobs
~~~~~~~~~~~~~~~~~~~

.. _task-listsdm:

listsdm
~~~~~~~~~~~~~~~~~~~

.. _task-mfsclean:

mfsclean
~~~~~~~~~~~~~~~~~~~

.. _task-mjd2date:

mjd2date
~~~~~~~~~~~~~~~~~~~

.. _task-mstransform:

mstransform
~~~~~~~~~~~~~~~~~~~

.. _task-plotants:

plotants
~~~~~~~~~~~~~~~~~~~

.. _task-plotcal:

plotcal
~~~~~~~~~~~~~~~~~~~

.. _task-setjy:

setjy
~~~~~~~~~~~~~~~~~~~

.. _task-split:

split
~~~~~~~~~~~~~~~~~~~

.. _task-tsysplot:

tsysplot
~~~~~~~~~~~~~~~~~~~

.. _task-uvsub:

uvsub
~~~~~~~~~~~~~~~~~~~

.. _task-xyphplot:

xyphplot
~~~~~~~~~~~~~~~~~~~
