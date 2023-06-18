.. Copyright 2015 Peter K. G. Williams <peter@newton.cx> and collaborators.
   This file licensed under the Creative Commons Attribution-ShareAlike 3.0
   Unported License (CC-BY-SA).

Numerical utilities (:mod:`pwkit.numutil`)
==============================================================================

.. automodule:: pwkit.numutil
   :synopsis: low-level numerical utilities

.. currentmodule:: pwkit.numutil

The functionality in this module can be grouped into these categories:

 - :ref:`auto-broadcasting`
 - :ref:`statistics`
 - :ref:`dataframes`
 - :ref:`parallelized`
 - :ref:`tophats-and-steps`


.. _auto-broadcasting:

Making functions that auto-broadcast their arguments
------------------------------------------------------------------------------

.. decorator:: broadcastize(n_arr, ret_spec=0, force_float=True)

   Wrap a function to automatically broadcast :class:`numpy.ndarray` arguments.

   It’s often desirable to write numerical utility functions in a way that’s
   compatible with vectorized processing. It can be tedious to do this,
   however, since the function arguments need to turned into arrays and
   checked for compatible shape, and scalar values need to be special cased.

   The ``@broadcastize`` decorator takes care of these matters. The decorated
   function can be implemented in vectorized form under the assumption that
   all array arguments have been broadcast to the same shape. The broadcasting
   of inputs and (potentially) de-vectorizing of the return values are done
   automatically. For instance, if you decorate a function ``foo(x,y)`` with
   ``@numutil.broadcastize(2)``, you can implement it assuming that both *x*
   and *y* are :class:`numpy.ndarray` objects that have at least one dimension
   and are both of the same shape. If the function is called with only scalar
   arguments, *x* and *y* will have shape ``(1,)`` and the function’s return
   value will be turned back into a scalar before reaching the caller.

   The *n_arr* argument specifies the number of array arguments that the
   function takes. These are required to be at the beginning of its argument
   list.

   The *ret_spec* argument specifies the structure of the function’s return
   value.

   - ``0`` indicates that the value has the same shape as the (broadcasted)
     vector arguments. If the arguments are all scalar, the return value will
     be scalar too.
   - ``1`` indicates that the value is an array of higher rank than the input
     arguments. For instance, if the input has shape ``(3,)``, the output
     might have shape ``(4,4,3)``; in general, if the input has shape ``s``,
     the output will have shape ``t + s`` for some tuple ``t``. If the
     arguments are all scalar, the output will have a shape of just ``t``. The
     :func:`numpy.asarray` function is called on such arguments, so (for
     instance) you can return a list of arrays ``[a, b]`` and it will be
     converted into a :class:`numpy.ndarray`.
   - ``None`` indicates that the value is completely independ of the inputs. It
     is returned as-is.
   - A tuple ``t`` indicates that the return value is also a tuple. The
     elements of the *ret_spec* tuple should contain the values listed above,
     and each element of the return value will be handled accordingly.

   The default *ret_spec* is ``0``, i.e. the return value is expected to be an
   array of the same shape as the argument(s).

   If *force_float* is true (the default), the input arrays will be converted to
   floating-point types if necessary (with :func:`numpy.asfarray`) before being
   passed to the function.

   Example::

     @numutil.broadcastize (2, ret_spec=(0, 1, None)):
     def myfunction (x, y, extra_arg):
         print ('a random non-vector argument is:', extra_arg)
	 z = x + y
	 z[np.where (y)] *= 2
	 higher_vector = [x, y, z]
	 return z, higher_vector, 'hello'


.. _statistics:

Convenience functions for statistics
------------------------------------------------------------------------------

.. autosummary::
   rms
   weighted_mean
   weighted_mean_df
   weighted_variance

.. autofunction:: rms
.. autofunction:: weighted_mean
.. autofunction:: weighted_mean_df
.. autofunction:: weighted_variance


.. _dataframes:

Convenience functions for :class:`pandas.DataFrame` objects
------------------------------------------------------------------------------

.. autosummary::
   reduce_data_frame
   reduce_data_frame_evenly_with_gaps
   slice_around_gaps
   slice_evenly_with_gaps
   dfsmooth
   smooth_data_frame_with_gaps
   fits_recarray_to_data_frame
   data_frame_to_astropy_table
   usmooth
   page_data_frame

.. autofunction:: reduce_data_frame
.. autofunction:: reduce_data_frame_evenly_with_gaps
.. autofunction:: slice_around_gaps
.. autofunction:: slice_evenly_with_gaps
.. autofunction:: dfsmooth
.. autofunction:: smooth_data_frame_with_gaps
.. autofunction:: fits_recarray_to_data_frame
.. autofunction:: data_frame_to_astropy_table
.. autofunction:: usmooth
.. autofunction:: page_data_frame


.. _parallelized:

Parallelized versions of simple math algorithms
------------------------------------------------------------------------------

.. autosummary::
   parallel_newton
   parallel_quad

.. autofunction:: parallel_newton
.. autofunction:: parallel_quad


.. _tophats-and-steps:

Tophat and step functions
------------------------------------------------------------------------------

.. autosummary::
   unit_tophat_ee
   unit_tophat_ei
   unit_tophat_ie
   unit_tophat_ii
   make_tophat_ee
   make_tophat_ei
   make_tophat_ie
   make_tophat_ii
   make_step_lcont
   make_step_rcont

.. autofunction:: unit_tophat_ee
.. autofunction:: unit_tophat_ei
.. autofunction:: unit_tophat_ie
.. autofunction:: unit_tophat_ii
.. autofunction:: make_tophat_ee
.. autofunction:: make_tophat_ei
.. autofunction:: make_tophat_ie
.. autofunction:: make_tophat_ii
.. autofunction:: make_step_lcont
.. autofunction:: make_step_rcont
