.. Copyright 2017 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Helpers for X-ray spectral modeling with the Sherpa packge (:mod:`pwkit.sherpa`)
================================================================================

.. automodule:: pwkit.sherpa
   :synopsis: helpers for X-ray spectral modeling with Sherpa

.. currentmodule:: pwkit.sherpa


This module includes a grab-bag of helpers in following broad topics:

 - :ref:`sherpa-models`
 - :ref:`sherpa-plots`
 - :ref:`sherpa-data-structure-utilities`


.. _sherpa-models:

Additional Spectral Models
--------------------------

The :mod:`pwkit.sherpa` module provides several tools for constructing models
not provided in the standard Sherpa distribution.

.. autoclass:: PowerLawApecDemModel

.. autofunction:: make_fixed_temp_multi_apec


.. _sherpa-plots:

Tools for Plotting with Sherpa Data Objects
-------------------------------------------

.. autosummary::
   get_source_qq_data
   get_bkg_qq_data
   make_qq_plot
   make_multi_qq_plots
   make_spectrum_plot
   make_multi_spectrum_plots


.. autofunction:: get_source_qq_data
.. autofunction:: get_bkg_qq_data
.. autofunction:: make_qq_plot
.. autofunction:: make_multi_qq_plots
.. autofunction:: make_spectrum_plot
.. autofunction:: make_multi_spectrum_plots


.. _sherpa-data-structure-utilities:

Data Structure Utilities
------------------------

.. autosummary::
   expand_rmf_matrix
   derive_identity_arf
   derive_identity_rmf


.. autofunction:: expand_rmf_matrix
.. autofunction:: derive_identity_arf
.. autofunction:: derive_identity_rmf
