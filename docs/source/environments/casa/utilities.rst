.. Copyright 2015-2017 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

CASA Tools and Utilities (:mod:`pwkit.environments.casa.util`)
================================================================================

.. automodule:: pwkit.environments.casa.util
   :synopsis: low-level CASA tools and utilities

.. currentmodule:: pwkit.environments.casa.util

This module provides:

 - :ref:`tools-object`
 - :ref:`useful-constants-casautil`
 - :ref:`useful-functions-casautil`


.. Note: I can't get ``autodata`` working; not sure why. Whatever.

.. _tools-object:

The ``tools`` object
--------------------

.. data:: tools
   :annotation:

   This object is a singleton instance of a hidden class that assists in the
   creation of CASA “tools” objects. For instance, you can create and use a
   standard CASA “tool” for reading and manipulating data tables with code
   like this::

     from pwkit.environments.casa import util
     tb = util.tools.table()
     tb.open('myfile.ms')
     tb.close()

   Documentation for the individual CASA “tools” is beyond the scope of
   :mod:`pwkit` … although maybe it will be added, since the documentation
   provided by CASA is pretty weak.

   Here’s a list of CASA tool names. They can all be created in the same way:
   by calling the function ``tools.<toolname>()``. This will work even for any
   tools not appearing in this list, so long as they’re provided by the
   underlying CASA libraries:

    - agentflagger
    - atmosphere
    - calanalysis
    - calibrater
    - calplot
    - componentlist
    - coordsys
    - deconvolver
    - fitter
    - flagger
    - functional
    - image
    - imagepol
    - imager
    - logsink
    - measures
    - msmetadata
    - ms
    - msplot
    - mstransformer
    - plotms
    - regionmanager
    - simulator
    - spectralline
    - quanta
    - table
    - tableplot
    - utils
    - vlafiller
    - vpmanager


.. _useful-constants-casautil:

Useful Constants
----------------

The following useful constants are provided:

.. data:: INVERSE_C_SM

   The inverse of the speed of light, *c*, measured in seconds per meter. This
   is useful for converting between wavelength and light travel time.

.. data:: INVERSE_C_NSM

   The inverse of the speed of light, *c*, measured in nanoseconds per meter.
   This is useful for converting between wavelength and light travel time.

.. data:: pol_names
   :annotation:

   A dictionary mapping CASA polarization codes to their textual names. For
   instance, ``pol_names[9]`` is ``"XX"`` and ``pol_names[7]`` is ``"LR"``.

.. data:: pol_to_miriad
   :annotation:

   A dictionary mapping CASA polarization codes to MIRIAD polarization codes,
   such that::

     miriad_pol_code = pol_to_miriad[casa_pol_code]

   CASA defines many more polarization codes than MIRIAD, although it is
   unclear whether CASA’s additional ones are ever used in practice. Trying to
   map a code without a MIRIAD equivalent will result in a :exc:`KeyError` as
   you might expect.

.. data:: pol_is_intensity
   :annotation:

   A dictionary mapping CASA polarization codes to booleans indicating whether
   the polarization is of “intensity” type. “Intensity-type” polarizations
   cannot have negative values; they are II, RR, LL, XX, YY, PP, and QQ.

.. data:: msselect_keys
   :annotation:

   A :class:`set` of the keys supported by the CASA “MS-select” subsystem.


.. _useful-functions-casautil:

Useful Functions
----------------

.. autosummary::
   sanitize_unicode
   datadir
   logger
   forkandlog

.. autofunction:: sanitize_unicode
.. autofunction:: datadir
.. autofunction:: logger
.. autofunction:: forkandlog
