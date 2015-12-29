.. Copyright 2015 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Framework for easy parallelized processing (``pwkit.parallel``)
==============================================================================

.. automodule:: pwkit.parallel
   :synopsis: Infrastructure for parallel processing

.. currentmodule:: pwkit.parallel


Main Interface
--------------

The most important parts of this module are the :func:`make_parallel_helper`
function and the interface defined by the abstract :class:`ParallelHelper`
class.

.. autosummary::

   make_parallel_helper
   ParallelHelper

.. autofunction:: make_parallel_helper

.. autoclass:: ParallelHelper

   .. automethod:: get_map
   .. automethod:: get_ppmap


Implementation Details
----------------------

Some of these classes and functions may be useful for other modules, but in
generally you need only concern yourself with the :func:`make_parallel_helper`
function and :class:`ParallelHelper` base class.

.. autosummary::

   SerialHelper
   serial_ppmap
   MultiprocessingPoolHelper
   multiprocessing_ppmap_worker
   InterruptiblePool
   VacuousContextManager

.. autoclass:: SerialHelper

.. autofunction:: serial_ppmap

.. autoclass:: MultiprocessingPoolHelper

.. autofunction:: multiprocessing_ppmap_worker

.. autoclass:: InterruptiblePool

.. autoclass:: VacuousContextManager
