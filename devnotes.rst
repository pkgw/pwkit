===============
Making releases
===============

1. Update version in ``setup.py``.
2. Register new version: ``python setup.py register``.
3. Upload files: ``python setup.py sdist bdist upload``.
4. Update ``meta.yaml`` in ``conda`` recipe.
5. Build for ``conda``: ``conda build .``.
6. Upload to ``binstar``: execute line at end of the ``conda build`` output.


=======================
Bootstrapping a project
=======================

1. Copy skeleton (e.g. this repository).
2. Pretty much do the steps for making a release.
3. Seed the ``conda`` recipe with: ``conda skeleton pypi <pkgname>``.
