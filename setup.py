# -*- coding: utf-8 -*-
# python setup.py install

import os
from setuptools import setup, find_packages

AUTHOR = 'Frank Longford'
AUTHOR_EMAIL = 'f.longford@soton.ac.uk'
URL = 'https://github.com/franklongford/ALIAS'
PLATFORMS = ['Linux', 'Unix', 'Mac OS X']
VERSION = '2.0.0'

PYTHON_MAJOR_VERSION = '3'
PYTHON_MINOR_VERSION = '6'


def write_version_py():

    filename = os.path.join(
        os.path.abspath('.'),
        'alias',
        'version.py')
    print(filename)
    ver = f"__version__ = '{VERSION}'\n"
    with open(filename, 'w') as outfile:
        outfile.write(ver)


write_version_py()

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt', 'r') as infile:
    REQUIREMENTS = infile.readlines()


setup(
    name='ALIAS',
    version=VERSION,
    description='Air-Liquid Interface Analysis Suite',
    long_description=readme,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    platforms=PLATFORMS,
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    python_requires=f'>={PYTHON_MAJOR_VERSION}.{PYTHON_MINOR_VERSION}',
    entry_points={
        'gui_scripts': ['ALIAS = alias.cli.main:alias']},
    install_requires=REQUIREMENTS
)
