.. Copyright 2015 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Basic astronomical calculations (:mod:`pwkit.astutil`)
========================================================================

.. module:: pwkit.astutil
   :synopsis: basic astronomical utilities

This module collects many utilities for performing basic astronomical calculations, including:

 - :ref:`useful-constants`
 - :ref:`sexagesimal`
 - :ref:`angles`
 - :ref:`gaussians`
 - :ref:`astrometry`


.. _useful-constants:

Useful Constants
------------------------------------------------------------------------------

.. data:: pi

   Placeholder.

.. data:: twopi

   Placeholder.

.. data:: halfpi

   Placeholder.

.. data:: R2A

   Placeholder.

.. data:: A2R

   Placeholder.

.. data:: R2D

   Placeholder.

.. data:: D2R

   Placeholder.

.. data:: R2H

   Placeholder.

.. data:: H2R

   Placeholder.

.. data:: F2S

   Placeholder.

.. data:: S2F

   Placeholder.

.. data:: J2000

   Placeholder.


.. _sexagesimal:

Sexagesimal Notation
------------------------------------------------------------------------------

.. function:: fmthours (radians, norm='wrap', precision=3, seps='::')

   Placeholder.

.. function:: fmtdeglon (radians, norm='wrap', precision=2, seps='::')

   Placeholder.

.. function:: fmtdeglat (radians, norm='raise', precision=2, seps='::')

   Placeholder.

.. function:: fmtradec (rarad, decrad, precision=2, raseps='::', decseps='::', intersep=' ')

   Placeholder.

.. function:: parsehours (hrstr)

   Placeholder.

.. function:: parsedeglat (latstr)

   Placeholder.

.. function:: parsedeglon (lonstr)

   Placeholder.


.. _angles:

Working with Angles
------------------------------------------------------------------------------

.. function:: angcen (a)

   Placeholder.

.. function:: orientcen (a)

   Placeholder.

.. function:: sphdist (lat1, lon1, lat2, lon2)

   Placeholder.

.. function:: sphbear (lat1, lon1, lat2, lon2, tol=1e-15)

   Placeholder.

.. function:: sphofs (lat1, lon1, r, pa, tol=1e-2, rmax=None)

   Placeholder.

.. function:: parang (hourangle, declination, latitude)

   Placeholder.


.. _gaussians:

Simple Operations on 2D Gaussians
------------------------------------------------------------------------------

.. function:: gaussian_convolve (maj1, min1, pa1, maj2, min2, pa2)

   Placeholder.

.. function:: gaussian_deconvolve (smaj, smin, spa, bmaj, bmin, bpa)

   Placeholder.


.. _astrometry:

Basic Astrometry
------------------------------------------------------------------------------

.. function:: get_2mass_epoch (tmra, tmdec, debug=False)

   Placeholder.

.. function:: get_simbad_astrometry_info (ident, items=..., debug=False)

   Placeholder.

.. class:: AstrometryInfo (simbadident=None, **kwargs)

   Placeholder.

.. method:: AstrometryInfo.verify (complain=True)

   Placeholder.

.. method:: AstrometryInfo.predict (mjd, complain=True, n=20000)

   Placeholder.

.. method:: AstrometryInfo.prin_prediction (ptup)

   Placeholder.

.. method:: AstrometryInfo.fill_from_simbad (ident, debug=False)

   Placeholder.
