#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for bikeshare_model package.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Package metadata
NAME = "bikeshare_model"
DESCRIPTION = "Bike sharing prediction model."
URL = "https://github.com/yourusername/bikeshare-model"
EMAIL = "arunv.inc@gmail.com"
AUTHOR = "Arun"
REQUIRES_PYTHON = ">=3.9.0"

# Load the package version
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
with open(PACKAGE_DIR / "VERSION") as f:
    VERSION = f.read().strip()
    about["__version__"] = VERSION

# Load the README file
with open(ROOT_DIR / "README.md") as readme_file:
    readme = readme_file.read()

# Define the required packages
def list_reqs(fname="requirements.txt"):
    with open(ROOT_DIR / fname) as fd:
        return [line.strip() for line in fd.read().splitlines() 
                if line.strip() and not line.startswith('#')]

# Setup configuration
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=readme,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"bikeshare_model": ["VERSION"]},
    install_requires=list_reqs(),
    # Then in the setup() call:
    #extras_require={
    #"dev": list_reqs("test_requirements.txt") if os.path.exists(ROOT_DIR / "test_requirements.txt") else [],
    #},
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Scientific/Engineering",
    ],
)