#! /usr/bin/env python
# Copyright 2014 Peter Williams <peter@newton.cx>
# Licensed under the MIT License.

# I don't use the ez_setup module because it causes us to automatically build
# and install a new setuptools module, which I'm not interested in doing.

from setuptools import setup

setup (
    name = 'pwkit',
    version = '0.1',

    zip_safe = True,
    packages = ['pwkit'],

    # install_requires = ['docutils >= 0.3'],

    # entry_points = {
    #     'console_scripts': ['bib = bibtools.cli:driver'],
    # },

    author = 'Peter Williams',
    author_email = 'peter@newton.cx',
    description = 'Miscellaneous scientific and astronomical tools',
    license = 'MIT',
    keywords = 'astronomy science',
    url = 'https://github.com/pkgw/pwkit/',
)
