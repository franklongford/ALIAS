"""
********************** TEST MODULE ************************

Main program, ALIAS: Air-Liquid Interface Analysis Suite

***********************************************************
Created 18/01/2018 by Frank Longford

Contributors: Frank Longford

Last modified 27/02/2018 by Frank Longford
"""

import numpy as np
import scipy as sp
import subprocess, time, sys, os, math, copy, gc, tables
import mdtraj as md

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import utilities as ut
import intrinsic_sampling_method as ism
import intrinsic_analysis as ia

THRESH = 1E-8


def test_unit_vector():

	vector = [-3, 2, 6]
	answer = np.array([-0.42857143,  0.28571429,  0.85714286])
	u_vector = ut.unit_vector(vector)

	assert np.sum(u_vector - answer) <= THRESH

	vector_array = [[3, 2, 6], [1, 2, 5], [4, 2, 5], [-7, -1, 2]]

	u_vector_array = ut.unit_vector(vector_array)

	assert np.array(vector_array).shape == u_vector_array.shape


def test_remove():

	array_1 = np.arange(50)
	array_2 = array_1 + 20
	answer = np.arange(20)

	edit_array = ut.numpy_remove(array_1, array_2)

	assert np.sum(answer - edit_array) <= THRESH


def test_checkfile():

	checkfile_name = 'test_checkfile'
	if not os.path.exists('{}.pkl'.format(checkfile_name)): ut.make_checkfile(checkfile_name)
	checkfile = ut.read_checkfile(checkfile_name)

	checkfile = ut.update_checkfile(checkfile_name, 'M', [16.0, 1.008, 1.008, 0])

	assert len(checkfile['M']) == 4
	assert np.sum(np.array(checkfile['M']) - 18.016) <= THRESH

	os.remove(checkfile_name + '.pkl')


def test_load_save():

	test_data = np.arange(50)
	test_name = 'test_load_save'

	ut.save_npy(test_name, test_data)
	load_data = ut.load_npy(test_name)

	assert abs(np.sum(test_data - load_data)) <= THRESH

	new_test_data = test_data[:10]
	load_data = ut.load_npy(test_name, frames=range(10))

	assert abs(np.sum(new_test_data - load_data)) <= THRESH

	os.remove(test_name + '.npy')

	ut.make_hdf5(test_name, test_data.shape, tables.Int64Atom())
	ut.save_hdf5(test_name, test_data, 0)
	
	assert ut.shape_check_hdf5(test_name) == (1,) + test_data.shape
	
	load_data = ut.load_hdf5(test_name, 0)

	assert abs(np.sum(test_data - load_data)) <= THRESH

	new_test_data = test_data * 20
	ut.save_hdf5(test_name, new_test_data, 0, mode='r+')
	load_data = ut.load_hdf5(test_name, 0)

	assert abs(np.sum(new_test_data - load_data)) <= THRESH

	os.remove(test_name + '.hdf5')


def test_molecules():

	xat = np.array([20.3155606, 20.3657056, 19.7335474, 20.2454104, 23.1171728, 23.0142095, 23.7594160, 23.1883006])
	yat = np.array([29.0287238, 28.0910350, 29.3759130, 28.9508404, 35.2457050, 34.8579738, 34.6865613, 35.1208178])
	zat = np.array([58.6756206, 58.8612466, 59.3516029, 58.7892616, 63.1022910, 63.9713681, 62.6651254, 63.1592576])
	mol_M = np.array([16.0, 1.008, 1.008, 0.0, 16.0, 1.008, 1.008, 0.0])

	answer_xmol = np.array([20.3155606,  23.1171728])
	answer_ymol = np.array([29.0287238,  35.2457050])
	answer_zmol = np.array([58.6756206,  63.1022910])

	xmol, ymol, zmol = ut.molecules(xat, yat, zat, 2, 4, mol_M, 0)

	assert xmol.shape == (2,)
	assert ymol.shape == (2,)
	assert zmol.shape == (2,)

	assert abs(np.sum(xmol - answer_xmol)) <= THRESH
	assert abs(np.sum(ymol - answer_ymol)) <= THRESH
	assert abs(np.sum(zmol - answer_zmol)) <= THRESH

	answer_xmol = np.array([20.28580243,  23.14734565])
	answer_ymol = np.array([28.99568519,  35.19272710])
	answer_zmol = np.array([58.72382781,  63.12645656])

	xmol, ymol, zmol = ut.molecules(xat, yat, zat, 2, 4, mol_M, 'COM')

	assert abs(np.sum(xmol - answer_xmol)) <= THRESH
	assert abs(np.sum(ymol - answer_ymol)) <= THRESH
	assert abs(np.sum(zmol - answer_zmol)) <= THRESH


def test_bubble_sort():

	array = np.array([0, 4, 3, 2, 7, 8, 1, 5, 6])
	key = np.array([0, 6, 3, 2, 1, 7, 8, 4, 5]) 
	answer = np.arange(9)

	ut.bubble_sort(array, key)
	
	assert abs(np.sum(array - answer)) <= THRESH
	assert abs(np.sum(key - answer)) <= THRESH


def test_wavefunctions():

	qm = 8
	qu = 5
	n_waves = 2 * qm + 1
	coeff = np.ones(n_waves**2) * 0.5
	dim = [10., 12.]

	pos = np.arange(10)
	u_array = np.arange(-qm, qm+1)

	xi_array = ism.xi(pos, pos, coeff, qm, qu, dim)
	dx_dxi_array, dy_dxi_array = ism.dxy_dxi(pos, pos, coeff, qm, qu, dim)
	ddx_ddxi_array, ddy_ddxi_array = ism.ddxy_ddxi(pos, pos, coeff, qm, qu, dim)
	H_array = ia.H_xy(pos, pos, coeff, qm, qu, dim)

	assert xi_array.shape == pos.shape
	assert dx_dxi_array.shape == pos.shape
	assert dy_dxi_array.shape == pos.shape
	assert ddx_ddxi_array.shape == pos.shape
	assert ddy_ddxi_array.shape == pos.shape
	assert H_array.shape == pos.shape

	for i, x in enumerate(pos):
		xi = ism.xi(x, x, coeff, qm, qu, dim)
		dx_dxi, dy_dxi = ism.dxy_dxi(x, x, coeff, qm, qu, dim)
		ddx_ddxi, ddy_ddxi = ism.ddxy_ddxi(x, x, coeff, qm, qu, dim)
		H = ia.H_xy(x, x, coeff, qm, qu, dim)

		assert abs(np.sum(xi - xi_array[i])) <= THRESH
		assert abs(np.sum(dx_dxi - dx_dxi_array[i])) <= THRESH
		assert abs(np.sum(dy_dxi - dy_dxi_array[i])) <= THRESH
		assert abs(np.sum(ddx_ddxi - ddx_ddxi_array[i])) <= THRESH
		assert abs(np.sum(ddy_ddxi - ddy_ddxi_array[i])) <= THRESH
		assert abs(np.sum(H - H_array[i])) <= THRESH

	pos = np.arange(100)
	H_array = ia.H_xy(pos, pos, coeff, qm, qu, dim)
	assert abs(ia.H_var_mol(pos, pos, coeff, qm, qu, dim) - np.var(H_array)) <= 0.07


def test_coeff_slice():

	qm = 5
	qu = 3
	n_waves = 2 * qm + 1

	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
	wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
	indices = np.argwhere(wave_check).flatten()

	"Create matrix of wave frequency indicies (u,v)**2"
	u_matrix_qm = np.tile(u_array, (n_waves, 1))
	v_matrix_qm = np.tile(v_array, (n_waves, 1))
	q_matrix_qm = np.sqrt(u_matrix_qm**2 + v_matrix_qm**2)

	u_matrix_qu = np.tile(u_array[indices], (len([indices]), 1))
	v_matrix_qu = np.tile(v_array[indices], (len([indices]), 1))
	q_matrix_qu = np.sqrt(u_matrix_qu**2 + v_matrix_qu**2)

	q_matrix_qu_2 = ia.coeff_slice(q_matrix_qm, qm, qu)

	assert abs(np.sum(q_matrix_qu_2 - q_matrix_qu)) <= THRESH


