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
package manager by installing it from `anaconda.org`_; the command ``conda
install -c pkgw pwkit`` should suffice.

If you want to download the source code and install ``pwkit`` manually, the
package uses the standard Python `setuptools`_, so running ``python setup.py
install`` will do the trick.

.. _Python package index: https://pypi.python.org/pypi/pwkit/
.. _conda: http://conda.pydata.org/docs/
.. _anaconda.org: https://anaconda.org/pkgw/pwkit
.. _setuptools: https://pypi.python.org/pypi/setuptools

Some ``pwkit`` functionality requires additional Python modules such as
`scipy`_; these issues should be very obvious as they manifest as
``ImportErrors`` triggered for the relevant modules. Bare minimum
functionality requires:

* `numpy`_ >= 1.6
* `six`_ >= 1.9
* on Python 2.x only, `pathlib`_ >= 1.0

If you install ``pwkit`` through standard means, these modules should be
automatically installed too if they weren’t already available.

.. _scipy: http://www.scipy.org/
.. _numpy: http://www.numpy.org/
.. _six: https://pythonhosted.org/six/
.. _pathlib: https://pypi.python.org/pypi/pathlib/


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
