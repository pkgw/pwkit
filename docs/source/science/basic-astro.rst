.. Copyright 2015 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Basic astronomical calculations (:mod:`pwkit.astutil`)
========================================================================

.. automodule:: pwkit.astutil
   :synopsis: basic astronomical utilities

.. currentmodule:: pwkit.astutil

This module collects many utilities for performing basic astronomical
calculations, including:

 - :ref:`useful-constants`
 - :ref:`sexagesimal`
 - :ref:`angles`
 - :ref:`gaussians`
 - :ref:`astrometry`

Angles are always measured in radians, whereas some other astronomical
codebases prefer degrees.


.. _useful-constants:

Useful Constants
------------------------------------------------------------------------------

The following useful constants are provided:

``pi``
  Mathematical π.
``twopi``
  Mathematical 2π.
``halfpi``
  Mathematical π/2.
``R2A``
  A constant for converting radians to arcseconds by multiplication::

    arcsec = radians * astutil.R2A

  Equal to ``3600 * 180 / pi`` or about 206265.
``A2R``
  A constant for converting arcseconds to radians by multiplication::

    radians = arcsec * astutil.A2R

``R2D``
  Analogous to ``R2A``: a constant for converting radians to degrees
``D2R``
  Analogous to ``A2R``: a constant for converting degrees to radians
``R2H``
  Analogous to ``R2A``: a constant for converting radians to hours
``H2R``
  Analogous to ``A2R``: a constant for converting hours to radians
``F2S``
  A constant for converting a Gaussian FWHM (full width at half maximum) to
  a standard deviation (σ) value by multiplication::

    sigma = fwhm * astutil.F2S

  Equal to ``(8 * ln(2))**-0.5`` or about 0.425.
``S2F``
  A constant for converting a Gaussian standard deviation (σ) value to a
  FWHM (full width at half maximum) by multiplication.
``J2000``
  The astronomical J2000.0 epoch as a MJD (modified Julian Date). Precisely
  equal to 51544.5.


.. _sexagesimal:

Sexagesimal Notation
------------------------------------------------------------------------------

.. autosummary::
   fmthours
   fmtdeglon
   fmtdeglat
   fmtradec
   parsehours
   parsedeglat
   parsedeglon

.. autofunction:: fmthours
.. autofunction:: fmtdeglon
.. autofunction:: fmtdeglat
.. autofunction:: fmtradec
.. autofunction:: parsehours
.. autofunction:: parsedeglat
.. autofunction:: parsedeglon


.. _angles:

Working with Angles
------------------------------------------------------------------------------

.. autosummary::
   angcen
   orientcen
   sphdist
   sphbear
   sphofs
   parang

.. function:: angcen(a)

   “Center” an angle *a* to be between -π and +π.

   Both *a* and the return value are, of course, in radians.

.. function:: orientcen(a)

   “Center” an orientation *a* to be between -π/2 and +π/2.

   Both *a* and the return value are, of course, in radians. An “orientation”
   is different than an angle because values that differ by just π, not 2π,
   are considered equivalent. Orientations can come up in the discussion of
   linear polarization, for example.

.. autofunction:: sphdist
.. autofunction:: sphbear
.. autofunction:: sphofs
.. autofunction:: parang


.. _gaussians:

Simple Operations on 2D Gaussians
------------------------------------------------------------------------------

.. autofunction:: gaussian_convolve
.. autofunction:: gaussian_deconvolve


.. _astrometry:

Basic Astrometry
------------------------------------------------------------------------------

.. autosummary::
   get_2mass_epoch
   get_simbad_astrometry_info

.. autoclass:: AstrometryInfo

.. automethod:: AstrometryInfo.verify
.. automethod:: AstrometryInfo.predict
.. automethod:: AstrometryInfo.print_prediction
.. automethod:: AstrometryInfo.fill_from_simbad
