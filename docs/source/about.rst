==================
About the Software
==================

``pwkit`` is a collection of Peter Williams’ miscellaneous Python tools. I’m
packaging them so that other people can install them off of PyPI or Conda and
run my code without having to go to too much work. That’s the hope, at least.


Installation
============

The most recent stable version of ``pwkit`` is available on the `Python
package index`_, so you should be able to install this package simply by
running ``pip install pwkit``. The package is also available in the `conda`_
package manager by installing it from `anaconda.org`_. If you are using
packages from the `conda-forge`_ project, install with ``conda install -c
pkgw-forge pwkit``. Otherwise, use ``conda install -c pkgw pwkit``.

If you want to download the source code and install ``pwkit`` manually, the
package uses the standard Python `setuptools`_, so running ``python setup.py
install`` will do the trick.

.. _Python package index: https://pypi.python.org/pypi/pwkit/
.. _conda: http://conda.pydata.org/docs/
.. _anaconda.org: https://anaconda.org/pkgw/pwkit
.. _conda-forge: http://conda-forge.github.io/
.. _setuptools: https://pypi.python.org/pypi/setuptools

Some ``pwkit`` functionality requires additional Python modules such as
`scipy`_; these issues should be very obvious as they manifest as
``ImportErrors`` triggered for the relevant modules. Bare minimum
functionality requires:

* `numpy`_ >= 1.6

If you install ``pwkit`` through standard means, these modules should be
automatically installed too if they weren’t already available.

.. _scipy: http://www.scipy.org/
.. _numpy: http://www.numpy.org/


Citation
========

.. Note: this text is mirrored in the toplevel README.rst

If you use pwkit in academic work, you should identify that you have done so
and specify the version used. While pwkit does not (yet?) have an accompanying
formal publication, in journals like `ApJ`_ you can “cite” the code directly via `its
record`_ in the `NASA Astrophysics Data System`_, which has identifier
`2017ascl.soft04001W`_. This corresponds to record `ascl:1704.001`_ in in the
`Astrophysics Source Code Library`_. By clicking on `this link`_ you can
get the ADS-recommended BibTeX record for the reference.

.. _ApJ: http://iopscience.iop.org/journal/0004-637X
.. _its record: https://ui.adsabs.harvard.edu/abs/2017ascl.soft04001W/abstract
.. _NASA Astrophysics Data System: https://ui.adsabs.harvard.edu/
.. _2017ascl.soft04001W: https://ui.adsabs.harvard.edu/abs/2017ascl.soft04001W/abstract
.. _ascl:1704.001: http://ascl.net/1704.001
.. _Astrophysics Source Code Library: http://ascl.net/
.. _this link: http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2017ascl.soft04001W&data_type=BIBTEX

If you are using `aastex`_ version 6 or higher, the appropriate code to
include after your Acknowledgments section would be:

.. _aastex: http://journals.aas.org/authors/aastex.html

.. code-block:: none

   \software{..., pwkit \citep{2017ascl.soft04001W}, ...}


Authors
=======

``pwkit`` is authored by Peter K. G. Williams and collaborators. Despite this
package being named after me, contributions are welcome and will be given full
credit. I just don’t want to have to make up a decent name for this package
right now.

Contributions have come from (alphabetically by surname):

* Maïca Clavel
* Elisabeth Newton
* Denis Ryzhkov (I copied `method_decorator`_)

.. _method_decorator: https://github.com/denis-ryzhkov/method_decorator/


Copyright and License
=====================

The ``pwkit`` package is copyright Peter K. G. Williams and collaborators and
licensed under the `MIT license`_, which is reproduced in the file LICENSE in
the source tree.

.. _MIT license: http://opensource.org/licenses/MIT
