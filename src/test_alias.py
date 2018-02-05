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

import utilities_edit as ut
import mdtraj as md

traj_dir = 'test'
traj_file = 'test_water.nc'
top_dir = 'test'
top_file = 'test_water.prmtop'

def test_sim_param():

	traj, MOL, ntraj, dim = ut.get_sim_param(traj_dir, top_dir, traj_file, top_file)

	assert ntraj == int(traj.n_frames)
	assert type(MOL) == list
	assert len(MOL) == 1
	assert MOL[0] == 'HOH'
	assert np.sum(dim - np.array([ 44.48912811,  44.1244812 ,  90.62278748])) <= 1E-5


def test_checkfile():

	file_name = traj_file.split('.')[0]
	ut.make_checkfile(traj_dir, file_name)
	checkfile = ut.read_checkfile(traj_dir, file_name)

	assert len(checkfile['M']) == 0
	assert checkfile['opt_ns'] == 0

	checkfile = ut.update_checkfile(traj_dir, file_name, 'M', [12, 1.0, 1.0, 1.0])

	assert len(checkfile['M']) == 4
	assert np.sum(checkfile['M']) == 15.0
