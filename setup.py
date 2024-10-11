#! /usr/bin/env python
# Copyright 2014-2023 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

# I don't use the ez_setup module because it causes us to automatically build
# and install a new setuptools module, which I'm not interested in doing.

from setuptools import setup


def get_long_desc():
    in_preamble = True
    lines = []

    with open("README.md", "rt", encoding="utf8") as f:
        for line in f:
            if in_preamble:
                if line.startswith("<!--pypi-begin-->"):
                    in_preamble = False
            else:
                if line.startswith("<!--pypi-end-->"):
                    break
                else:
                    lines.append(line)

    lines.append(
        """

For more information, including installation instructions, please visit [the
project homepage].

[the project homepage]: https://pwkit.readthedocs.io/
"""
    )
    return "".join(lines)


setup(
    name="pwkit",  # cranko project-name
    version="1.2.2",  # cranko project-version
    # This package actually *is* zip-safe, but I've run into issues with
    # installing it as a Zip: in particular, the install sometimes fails with
    # "bad local file header", and reloading a module after a reinstall in
    # IPython gives an ImportError with the same message. These are annoying
    # enough and I don't really care so we just install it as flat files.
    zip_safe=False,
    packages=[
        "pwkit",
        "pwkit.cli",
        "pwkit.environments",
        "pwkit.environments.casa",
        "pwkit.environments.ciao",
        "pwkit.environments.jupyter",
        "pwkit.environments.sas",
    ],
    package_data={
        "pwkit": ["data/*/*"],
    },
    # We want to go easy on the requires; some modules are going to require
    # more stuff, but others don't need much of anything. But, it's pretty
    # much impossible to do science without Numpy. We may actually require
    # version 1.8 but for now I'll optimistically hope that we can get away
    # with 1.6.
    install_requires=[
        "numpy >= 1.6",
    ],
    extras_require={
        "docs": [
            "mock",
            "numpydoc",
            "sphinx",
            "sphinx_rtd_theme",
            "sphinx-automodapi",
        ],
    },
    entry_points={
        "console_scripts": [
            "astrotool = pwkit.cli.astrotool:commandline",
            "casatask = pwkit.environments.casa.tasks:commandline",
            "imtool = pwkit.cli.imtool:commandline",
            "latexdriver = pwkit.cli.latexdriver:commandline",
            "pkcasascript = pwkit.environments.casa.scripting:commandline",
            "pkenvtool = pwkit.environments:commandline",
            "wrapout = pwkit.cli.wrapout:commandline",
        ],
    },
    author="Peter Williams",
    author_email="peter@newton.cx",
    description="Miscellaneous scientific and astronomical tools",
    license="MIT",
    keywords="astronomy science",
    url="https://github.com/pkgw/pwkit/",
    long_description=get_long_desc(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
