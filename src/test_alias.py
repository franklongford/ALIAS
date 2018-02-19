"""
********************** TEST MODULE ************************

Main program, ALIAS: Air-Liquid Interface Analysis Suite

***********************************************************
Created 18/01/2018 by Frank Longford

Contributors: Frank Longford

Last modified 18/01/2018 by Frank Longford
"""

import numpy as np
import scipy as sp
import subprocess, time, sys, os, math, copy, gc

import utilities as ut
import mdtraj as md

traj_dir = '/home/fl7g13/test'
traj_file = 'test_water.nc'
top_dir = '/home/fl7g13/test'
top_file = 'test_water.prmtop'

alias_dir = traj_dir + '/alias_analysis'

def test_unit_vector():

	vector = [-3, 2, 6]
	u_vector = ut.unit_vector(vector)

	assert np.sum(u_vector - np.array([-0.42857143,  0.28571429,  0.85714286])) <= 1E-5

	vector_array = [[3, 2, 6], [1, 2, 5], [4, 2, 5], [-7, -1, 2]]

	u_vector_array = ut.unit_vector(vector_array)

	assert np.array(vector_array).shape == u_vector_array.shape


def test_sim_param():

	traj, MOL, ntraj, dim = ut.get_sim_param(traj_dir, top_dir, traj_file, top_file)

	assert ntraj == int(traj.n_frames)
	assert type(MOL) == list
	assert len(MOL) == 1
	assert MOL[0] == 'HOH'
	assert np.sum(dim - np.array([ 44.48912811,  44.1244812 ,  90.62278748])) <= 1E-5


def test_checkfile():

	checkfile_name = '{}/{}_chk'.format(alias_dir, traj_file.split('.')[0])
	if not os.path.exists('{}.pkl'.format(checkfile_name)): ut.make_checkfile(checkfile_name)
	checkfile = ut.read_checkfile(checkfile_name)

	checkfile = ut.update_checkfile(checkfile_name, 'M', [16.0, 1.008, 1.008, 0])

	assert len(checkfile['M']) == 4
	assert np.sum(np.array(checkfile['M']) - 18.016) <= 1E-5
