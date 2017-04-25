.. Copyright 2017 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Helpers for X-ray spectral modeling with the Sherpa packge (:mod:`pwkit.sherpa`)
================================================================================

.. automodule:: pwkit.sherpa
   :synopsis: helpers for X-ray spectral modeling with Sherpa

.. currentmodule:: pwkit.sherpa


This module includes a grab-bag of helpers in following broad topics:

 - :ref:`sherpa-plots`
 - :ref:`sherpa-data-structure-utilities`
 - :ref:`partial-response-model-workaround`


.. _sherpa-plots:

Tools for Plotting with Sherpa Data Objects
-------------------------------------------

.. autosummary::
   get_source_qq_data
   get_bkg_qq_data
   make_qq_plot
   make_spectrum_plot


.. autofunction:: get_source_qq_data
.. autofunction:: get_bkg_qq_data
.. autofunction:: make_qq_plot
.. autofunction:: make_spectrum_plot


.. _sherpa-data-structure-utilities:

Data Structure Utilities
------------------------

.. autosummary::
   expand_rmf_matrix


.. autofunction:: expand_rmf_matrix


.. _partial-response-model-workaround:

Workarounds for models that only partially obey the instrumental response
-------------------------------------------------------------------------

.. autoclass:: FilterAdditionHack
