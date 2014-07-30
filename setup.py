#! /usr/bin/env python
# Copyright 2014 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

# I don't use the ez_setup module because it causes us to automatically build
# and install a new setuptools module, which I'm not interested in doing.

from setuptools import setup

setup (
    name = 'pwkit',
    version = '0.1.1',

    zip_safe = True,
    packages = [
        'pwkit',
        'pwkit.cli',
    ],

    package_data = {
        'pwkit': ['data/*/*'],
    },

    # We want to go easy on the requires; some modules are going to require
    # more stuff, but others don't need much of anything. But, it's pretty
    # much impossible to do science without Numpy. We may actually require
    # version 1.8 but for now I'll optimistically hope that we can get away
    # with 1.6.
    install_requires = [
        'numpy >= 1.6',
    ],

    entry_points = {
        'console_scripts': [
            'astrotool = pwkit.cli.astrotool:commandline',
            'imtool = pwkit.cli.imtool:commandline',
            'latexdriver = pwkit.cli.latexdriver:commandline',
            'wrapout = pwkit.cli.wrapout:commandline',
        ],
    },

    author = 'Peter Williams',
    author_email = 'peter@newton.cx',
    description = 'Miscellaneous scientific and astronomical tools',
    license = 'MIT',
    keywords = 'astronomy science',
    url = 'https://github.com/pkgw/pwkit/',

    long_description = \
    '''This is a collection of Peter Williams' miscellaneous Python tools. I'm
    packaging them so that other people can install them off of PyPI and run
    my code without having to go to too much work. That's the hope, at least.
    ''',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)
