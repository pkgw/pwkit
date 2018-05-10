.. Copyright 2015-2018 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Fitting generic models with least-squares minimization (:mod:`pwkit.lsqmdl`)
============================================================================

.. automodule:: pwkit.lsqmdl
   :synopsis: Fitting generic models with least-squares minimization

.. currentmodule:: pwkit.lsqmdl

There are four basic approaches all offering a common programming interface:

 - :ref:`generic-nonlinear`
 - :ref:`polynomial`
 - :ref:`scale-factor`
 - :ref:`component-based`

.. autosummary::
   ModelBase
   Parameter

.. autoclass:: ModelBase
   :members:
.. autoclass:: Parameter
   :members:


.. _generic-nonlinear:

Generic Nonlinear Modeling
--------------------------

.. autosummary::
   Model
   Parameter

.. autoclass:: Model
   :members:


.. _polynomial:

One-dimensional Polynomial Modeling
-----------------------------------

.. autoclass:: PolynomialModel
   :members:


.. _scale-factor:

Modeling of a Single Scale Factor
---------------------------------

.. autoclass:: ScaleModel
   :members:


.. _component-based:

Modeling With Pluggable Components
----------------------------------

.. autosummary::
   ComposedModel
   ModelComponent
   AddConstantComponent
   AddValuesComponent
   AddPolynomialComponent
   SeriesComponent
   MatMultComponent
   ScaleComponent

.. autoclass:: ComposedModel
   :members:
.. autoclass:: ModelComponent
   :members:
.. autoclass:: AddConstantComponent
   :members:
.. autoclass:: AddValuesComponent
   :members:
.. autoclass:: AddPolynomialComponent
   :members:
.. autoclass:: SeriesComponent
   :members:
.. autoclass:: MatMultComponent
   :members:
.. autoclass:: ScaleComponent
   :members:
