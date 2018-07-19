# -*- coding: utf-8 -*-
# python setup.py install

from setuptools import setup, find_packages

AUTHOR = 'Frank Longford'
AUTHOR_EMAIL = 'f.longford@soton.ac.uk'
URL = 'https://github.com/franklongford/ALIAS'
PLATFORMS = ['Linux', 'Unix', 'Mac OS X']
PACKAGE_DIR = {'ALIAS/src': '.'}

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


setup(
	name='ALIAS',
	version='1.2.0.dev1',
	description='Air-Liquid Interface Analysis Suite',
	long_description=readme,
	author=AUTHOR,
	author_email=AUTHOR_EMAIL,
	url=URL,
	platforms=PLATFORMS,
	package_dir=PACKAGE_DIR,
	license=license,
	packages=find_packages(exclude=('tests', 'docs')),
	python_requires='>=2.7'
)

