#!/usr/bin/env python

"""
# Setup module for BELKA
"""

import os
import sys

from setuptools import setup, find_packages

sys.path.insert(0, f"{os.path.dirname(__file__)}/BELKA")

import BELKA

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)), "BELKA")

setup(
    name="BELKA",
    entry_points={
        "console_scripts": [
            "BELKA = BELKA.__main__:main",
        ],
    },
    packages=find_packages(),
    version=BELKA.__version__,
    install_requires=[
        "scipy",
        "numpy",
        "pandas",
        "matplotlib",
        "pyvis",
        "jupyter",
        "networkx",
        "ipykernel",
        "scikit-learn",
        "torch"
    ]
)
