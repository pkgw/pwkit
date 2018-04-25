.. Copyright 2015-2018 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Synthetic photometry (``pwkit.synphot``)
==============================================================================

.. automodule:: pwkit.synphot
   :synopsis: synthetic photometry

.. currentmodule:: pwkit.synphot


The Registry class
------------------

.. autofunction:: get_std_registry

.. autoclass:: Registry

Instances of :class:`Registry` have the following methods:

.. autosummary::
   ~Registry.bands
   ~Registry.get
   ~Registry.register_bpass
   ~Registry.register_halfmaxes
   ~Registry.register_pivot_wavelength
   ~Registry.telescopes

.. automethod:: Registry.bands
.. automethod:: Registry.get
.. automethod:: Registry.register_bpass
.. automethod:: Registry.register_halfmaxes
.. automethod:: Registry.register_pivot_wavelength
.. automethod:: Registry.telescopes

.. autodata:: builtin_registrars


The Bandpass class
------------------

.. autoclass:: Bandpass

Instances of :class:`Bandpass` have the following attributes:

.. autosummary::
   ~Bandpass.band
   ~Bandpass.native_flux_kind
   ~Bandpass.registry
   ~Bandpass.telescope

And the following methods:

.. autosummary::
   ~Bandpass.calc_halfmax_points
   ~Bandpass.calc_pivot_wavelength
   ~Bandpass.halfmax_points
   ~Bandpass.jy_to_flam
   ~Bandpass.mag_to_flam
   ~Bandpass.mag_to_fnu
   ~Bandpass.pivot_wavelength
   ~Bandpass.synphot
   ~Bandpass.blackbody

.. rubric:: Detailed descriptions of attributes

.. autoattribute:: Bandpass.band
.. autoattribute:: Bandpass.native_flux_kind
.. autoattribute:: Bandpass.registry
.. autoattribute:: Bandpass.telescope

.. rubric:: Detailed descriptions of methods

.. automethod:: Bandpass.calc_halfmax_points
.. automethod:: Bandpass.calc_pivot_wavelength
.. automethod:: Bandpass.halfmax_points
.. automethod:: Bandpass.jy_to_flam
.. automethod:: Bandpass.mag_to_flam
.. automethod:: Bandpass.mag_to_fnu
.. automethod:: Bandpass.pivot_wavelength
.. automethod:: Bandpass.synphot
.. automethod:: Bandpass.blackbody


Simple, careful conversions
---------------------------

.. autosummary::
   fnu_cgs_to_flam_ang
   flam_ang_to_fnu_cgs
   abmag_to_fnu_cgs
   abmag_to_flam_ang
   ghz_to_ang
   flat_ee_bandpass_pivot_wavelength
   pivot_wavelength_ee
   pivot_wavelength_qe

.. autofunction:: fnu_cgs_to_flam_ang
.. autofunction:: flam_ang_to_fnu_cgs
.. autofunction:: abmag_to_fnu_cgs
.. autofunction:: abmag_to_flam_ang
.. autofunction:: ghz_to_ang
.. autofunction:: flat_ee_bandpass_pivot_wavelength
.. autofunction:: pivot_wavelength_ee
.. autofunction:: pivot_wavelength_qe


Exceptions
----------

.. autosummary::
   AlreadyDefinedError
   NotDefinedError

.. autoclass:: AlreadyDefinedError
.. autoclass:: NotDefinedError
