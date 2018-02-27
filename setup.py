# -*- coding: utf-8 -*-
# python setup.py install

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


setup(
	name='ALIAS',
	version='1.2.0.dev1',
	description='Intinsic Surface Analysis tools for simulations of air-liquid interfaces',
	long_description=readme,
	author='Frank Longford',
	author_email='f.longford@soton.ac.uk',
	url='https://github.com/franklongford/alias',
	license=license,
	packages=find_packages(exclude=('tests', 'docs')),
	python_requires='>=2.7'
)

