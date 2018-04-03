.. Copyright 2018 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Simple synchrotron radiation emission coefficients (:mod:`pwkit.dulk_models`)
=============================================================================

.. automodule:: pwkit.dulk_models
   :synopsis: simple synchrotron radiation emission coefficients

.. currentmodule:: pwkit.dulk_models

The models are from Dulk (1985; 1985ARA&A..23..169D;
doi:10.1146/annurev.aa.23.090185.001125). There are three versions:

 - :ref:`free-free`
 - :ref:`gyrosynchrotron`
 - :ref:`relativistic-synchrotron`
 - :ref:`helpers`

.. _free-free:

Free-free emission
------------------

.. autosummary::
   calc_freefree_kappa
   calc_freefree_eta
   calc_freefree_snu_ujy

.. autofunction:: calc_freefree_kappa
.. autofunction:: calc_freefree_eta
.. autofunction:: calc_freefree_snu_ujy


.. _gyrosynchrotron:

Gyrosynchrotron emission
------------------------

.. autosummary::
   calc_gs_kappa
   calc_gs_eta
   calc_gs_snu_ujy

.. autofunction:: calc_gs_kappa
.. autofunction:: calc_gs_eta
.. autofunction:: calc_gs_snu_ujy


.. _relativistic-synchrotron:

Relativistic synchrotron emission
---------------------------------

.. autosummary::
   calc_synch_kappa
   calc_synch_eta
   calc_synch_snu_ujy

.. autofunction:: calc_synch_kappa
.. autofunction:: calc_synch_eta
.. autofunction:: calc_synch_snu_ujy


.. _helpers:

Helpers
-------

.. autosummary::
   calc_nu_b
   calc_snu

.. autofunction:: calc_nu_b
.. autofunction:: calc_snu
