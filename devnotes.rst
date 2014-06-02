===============
Making releases
===============

1. Update version in ``setup.py``.
2. Build, register, upload: ``python setup.py sdist bdist register upload``.
3. Update ``meta.yaml`` in ``conda`` recipe.
4. Build for ``conda``: ``conda build .``.
5. Upload to ``binstar``: execute line at end of the ``conda build`` output.


=======================
Bootstrapping a project
=======================

1. Copy skeleton (e.g. this repository).
2. Pretty much do the steps for making a release.
3. Seed the ``conda`` recipe with: ``conda skeleton pypi <pkgname>``.
