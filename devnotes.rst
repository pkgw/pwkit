===============
Making releases
===============

1. Update version in ``setup.py``, ``pwkit/__init__.py``, ``docs/source/conf.py``.
2. Build, register, upload: ``python setup.py sdist bdist register upload``.
3. Update version again to ``X.Y.0.99``.
