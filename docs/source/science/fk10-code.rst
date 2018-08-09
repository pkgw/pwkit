.. Copyright 2018 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Run the Fleischman & Kuznetsov (2010) synchrotron code (:mod:`pwkit.fk10`)
==========================================================================

.. automodule:: pwkit.fk10
   :synopsis: run the Fleischman & Kuznetsov (2010) synchrotron code

.. currentmodule:: pwkit.fk10

.. autoclass:: Calculator

   .. rubric:: Setting parameters

   .. autosummary::
      set_bfield
      set_bfield_for_s0
      set_edist_powerlaw
      set_edist_powerlaw_gamma
      set_freqs
      set_hybrid_parameters
      set_ignore_q_terms
      set_obs_angle
      set_one_freq
      set_padist_gaussian_loss_cone
      set_padist_isotropic
      set_thermal_background
      set_trapezoidal_integration

   .. rubric:: Running calculations

   .. autosummary::
      find_rt_coefficients
      find_rt_coefficients_tot_intens

   .. automethod:: set_bfield
   .. automethod:: set_bfield_for_s0
   .. automethod:: set_edist_powerlaw
   .. automethod:: set_edist_powerlaw_gamma
   .. automethod:: set_freqs
   .. automethod:: set_hybrid_parameters
   .. automethod:: set_ignore_q_terms
   .. automethod:: set_obs_angle
   .. automethod:: set_one_freq
   .. automethod:: set_padist_gaussian_loss_cone
   .. automethod:: set_padist_isotropic
   .. automethod:: set_thermal_background
   .. automethod:: set_trapezoidal_integration

   .. automethod:: find_rt_coefficients
   .. automethod:: find_rt_coefficients_tot_intens
