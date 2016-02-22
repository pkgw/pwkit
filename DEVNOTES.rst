===============
Making releases
===============

1. Update version in ``setup.py``, ``pwkit/__init__.py``, ``docs/source/conf.py``; commit
2. Build, register, upload: ``python setup.py sdist bdist register upload``.
3. Tag as ``vX.Y.Z``; push tag with ``git push --tags``.
4. Update version again to ``X.Y.Z.99``; commit
