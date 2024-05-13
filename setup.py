#!/usr/bin/env python3

from setuptools import setup  # type: ignore


def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]


setup(
    name="bitnet_vae",
    version="0.1.0",
    description="bachelorthesis",
    author="Tarek Akrout",
    long_description_content_type="text/markdown",
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.10.12",
)
