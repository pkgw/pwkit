.. Copyright 2015 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Basic astronomical calculations (:mod:`pwkit.astutil`)
========================================================================

.. automodule:: pwkit.astutil
   :synopsis: basic astronomical utilities

.. currentmodule:: pwkit.astutil

This topics covered in this module are:

 - :ref:`useful-constants`
 - :ref:`sexagesimal`
 - :ref:`angles`
 - :ref:`gaussians`
 - :ref:`astrometry`
 - :ref:`misc-astronomy`

Angles are **always** measured in radians, whereas some other astronomical
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

   This is done by adding or subtracting multiples of 2π as necessary. Both
   *a* and the return value are in radians. The argument may be a vector.

.. function:: orientcen(a)

   “Center” an orientation *a* to be between -π/2 and +π/2.

   This is done by adding or subtract multiples of π as necessary. Both *a*
   and the return value are in radians. The argument may be a vector.

   An “orientation” is different than an angle because values that differ by
   just π, not 2π, are considered equivalent. Orientations can come up in the
   discussion of linear polarization, for example.

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

The :class:`AstrometryInfo` class can be used to perform basic astrometric
calculations that are nonetheless fairly accurate.

.. autoclass:: AstrometryInfo

   The attributes encoding the astrometric data are as follows. Values of
   ``None`` will be treated as unknown. Most of this information can be
   automatically filled in from the :meth:`fill_from_simbad` function, if you
   trust Simbad.

   .. autosummary::
      ra
      dec
      pos_u_maj
      pos_u_min
      pos_u_pa
      pos_epoch
      promo_ra
      promo_dec
      promo_u_maj
      promo_u_min
      promo_u_pa
      parallax
      u_parallax
      vradial
      u_vradial

   Methods are:

   .. autosummary::
      verify
      predict
      print_prediction
      predict_without_uncertainties
      fill_from_simbad
      fill_from_allwise

   The stringification of an :class:`AstrometryInfo` class formats its fields
   in a human-readable, multiline format that uses Unicode characters.

.. autoattribute:: AstrometryInfo.ra
.. autoattribute:: AstrometryInfo.dec
.. autoattribute:: AstrometryInfo.pos_u_maj
.. autoattribute:: AstrometryInfo.pos_u_min
.. autoattribute:: AstrometryInfo.pos_u_pa
.. autoattribute:: AstrometryInfo.pos_epoch
.. autoattribute:: AstrometryInfo.promo_ra
.. autoattribute:: AstrometryInfo.promo_dec
.. autoattribute:: AstrometryInfo.promo_u_maj
.. autoattribute:: AstrometryInfo.promo_u_min
.. autoattribute:: AstrometryInfo.promo_u_pa
.. autoattribute:: AstrometryInfo.parallax
.. autoattribute:: AstrometryInfo.u_parallax
.. autoattribute:: AstrometryInfo.vradial
.. autoattribute:: AstrometryInfo.u_vradial
.. automethod:: AstrometryInfo.verify
.. automethod:: AstrometryInfo.predict
.. automethod:: AstrometryInfo.print_prediction
.. automethod:: AstrometryInfo.predict_without_uncertainties
.. automethod:: AstrometryInfo.fill_from_simbad
.. automethod:: AstrometryInfo.fill_from_allwise

A few helper functions may also be of interest:

.. autosummary::
   load_skyfield_data
   get_2mass_epoch
   get_simbad_astrometry_info

.. autofunction:: load_skyfield_data
.. autofunction:: get_2mass_epoch
.. autofunction:: get_simbad_astrometry_info


.. _misc-astronomy:

Miscellaneous Astronomical Computations
------------------------------------------------------------------------------

These functions don’t fit under the other rubrics very well.

.. autosummary::
   abs2app
   app2abs

.. autofunction:: abs2app
.. autofunction:: app2abs
