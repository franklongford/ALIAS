"""
***************** INTRINSIC ANALYSIS MODULE *********************

Calculates properties of intrinsic surfaces, based on output files of
intrinsic_surface_method.py

********************************************************************
Created 22/2/2018 by Frank Longford

Contributors: Frank Longford

Last modified 27/2/2018 by Frank Longford
"""

import numpy as np
import scipy as sp, scipy.constants as con

import utilities as ut
import intrinsic_sampling_method as ism

import os, sys, time, tables


vcheck = np.vectorize(ism.check_uv)


def av_intrinsic_dist(directory, file_name, dim, nslice, qm, n0, phi, nframe, nsample, nz=100, recon=False, ow_dist=False):
	"""
	av_intrinsic_dist(directory, file_name, dim, nslice, qm, n0, phi, nframe, nsample, nz=100, recon=False, ow_dist=False)

	Summate average density and curvature distributions

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed.
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	nslice: int
		Number of bins in density histogram along axis normal to surface
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	n0:  int
		Maximum number of molecular pivots in intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	nframe:  int
		Number of frames in simulation trajectory
	nsample:  int
		Number of frames to average over
	nz: int (optional)
		Number of bins in curvature histogram along axis normal to surface (default=100)
	recon:  bool (optional)
		Whether to use surface reconstructe coefficients (default=False)
	ow_dist:  bool (optional)
		Whether to overwrite average density and curvature distributions (default=False)

	Returns
	-------

	int_den_curve_matrix:  float, array_like; shape=(qm+1, nslice, nz)
		Average intrinsic density-curvature distribution for each resolution across nsample frames
	int_density:  float, array_like; shape=(qm+1, nslice)
		Average intrinsic density distribution for each resolution across nsample frames
	int_curvature:  float, array_like; shape=(qm+1, nz)
		Average intrinsic surface curvature distribution for each resolution across nsample frames

	"""
	
	intden_dir = directory + 'intden/'

	file_name_count = '{}_{}_{}_{}_{}_{}_{}'.format(file_name, nslice, nz, qm, n0, int(1./phi + 0.5), nframe)

	if recon: file_name_count += '_R'

	if not os.path.exists(intden_dir + file_name_count + '_int_den_curve.npy'):

		int_den_curve_matrix = np.zeros((qm+1, nslice, nz))

		print "\n--- Loading in Density and Curvature Distributions ---\n"

		lslice = dim[2] / nslice
		Vslice = dim[0] * dim[1] * lslice

		for frame in xrange(nsample):
			sys.stdout.write("Frame {}\r".format(frame))
			sys.stdout.flush()

			count_corr_array = ut.load_hdf5(intden_dir + file_name_count + '_count_corr', frame)
			int_den_curve_matrix += count_corr_array / (nsample * Vslice)

		ut.save_npy(intden_dir + file_name_count + '_int_den_curve', int_den_curve_matrix)

	else:
		int_den_curve_matrix = ut.load_npy(intden_dir + file_name_count + '_int_den_curve')

	int_density = np.sum(int_den_curve_matrix, axis=2) / 2.
	int_curvature = np.sum(np.moveaxis(int_den_curve_matrix, 1, 2), axis=2) / 2.

	return int_den_curve_matrix, int_density, int_curvature


def H_xy(x, y, coeff, qm, qu, dim):
	"""
	H_xy(x, y, coeff, qm, qu, dim)

	Calculation of mean curvature at position (x,y) at resolution qu

	Parameters
	----------

	x:  float
		Coordinate in x dimension
	y:  float
		Coordinate in y dimension
	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	H:  float
		Mean curvature of intrinsic surface at point x,y
	"""

	n_waves = 2 * qm + 1

	if np.isscalar(x):
		u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
		v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
		wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
		indices = np.argwhere(wave_check).flatten()

		fuv = ism.wave_function_array(x, u_array[indices], dim[0]) * ism.wave_function_array(y, v_array[indices], dim[1])
		H = -4 * np.pi**2 * np.sum((u_array[indices]**2 / dim[0]**2 + v_array[indices]**2 / dim[1]**2) * fuv * coeff[indices])
	else:
		H_array = np.zeros(x.shape)
		for u in xrange(-qu, qu+1):
			for v in xrange(-qu, qu+1):
				j = (2 * qm + 1) * (u + qm) + (v + qm)
				H_array += ism.wave_function(x, u, dim[0]) * ism.wave_function(y, v, dim[1]) * (u**2 / dim[0]**2 + v**2 / dim[1]**2) * coeff[j]
		H = -4 * np.pi**2 * H_array

	return H


def H_var_coeff(coeff_2, qm, qu, dim):
	"""
	H_var_coeff(coeff_2, qm, qu, dim)

	Variance of mean curvature H across surface determined by coeff at resolution qu

	Parameters
	----------

	coeff_2:  float, array_like; shape=(n_waves**2)
		Square of optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	H_var:  float
		Variance of mean curvature H across whole surface

	"""

	if qu == 0: return 0
	
	n_waves = 2 * qm +1
	
	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
	wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
	indices = np.argwhere(wave_check).flatten()

	H_var_array = vcheck(u_array[indices], v_array[indices]) * coeff_2[indices]
	H_var_array *= (u_array[indices]**4 / dim[0]**4 + v_array[indices]**4 / dim[1]**4 + 2 * u_array[indices]**2 * v_array[indices]**2 / (dim[0]**2 * dim[1]**2))
	H_var = 4 * np.pi**4 * np.sum(H_var_array) 

	return H_var


def H_var_mol(xmol, ymol, coeff, qm, qu, dim):
	"""
	H_var_mol(xmol, ymol, coeff, pivot, qm, qu, dim)

	Variance of mean curvature H at molecular positions determined by coeff at resolution qu

	Parameters
	----------

	xmol:  float, array_like; shape=(nmol)
		Molecular coordinates in x dimension
	ymol:  float, array_like; shape=(nmol)
		Molecular coordinates in y dimension
	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	H_var:  float
		Variance of mean curvature H at pivot points

	"""

	if qu == 0: return 0
	
	n_waves = 2 * qm +1
	nmol = xmol.shape[0]

	"Create arrays of wave frequency indicies u and v"
	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
	wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
	indices = np.argwhere(wave_check).flatten()

	"Create matrix of wave frequency indicies (u,v)**2"
	u_matrix = np.tile(u_array[indices], (len([indices]), 1))
	v_matrix = np.tile(v_array[indices], (len([indices]), 1))

	"Make curvature diagonal terms of A matrix"
	curve_diag = 16 * np.pi**4 * (u_matrix**2 * u_matrix.T**2 / dim[0]**4 + v_matrix**2 * v_matrix.T**2 / dim[1]**4 +
				     (u_matrix**2 * v_matrix.T**2 + u_matrix.T**2 * v_matrix**2) / (dim[0]**2 * dim[1]**2))

	"Form the diagonal xi^2 terms and b vector solutions"
        fuv = np.zeros((n_waves**2, nmol))
        for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
                	fuv[j] = ism.wave_function(xmol, u_array[j], dim[0]) * ism.wave_function(ymol, v_array[j], dim[1])
	ffuv = np.dot(fuv[indices], fuv[indices].T)

	coeff_matrix = np.tile(coeff[indices], (len([indices]), 1))
	H_var = np.sum(coeff_matrix * coeff_matrix.T * ffuv * curve_diag / nmol)

	return H_var


def get_frequency_set(qm, dim):
	"""
	get_frequency_set(qm, dim)

	Returns set of unique frequencies in Fouier series

	Parameters
	----------

	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------
	
	q_set:  float, array_like
		Set of unique frequencies
	q2_set:  float, array_like
		Set of unique frequencies to bin coefficients to
	"""

	q_set = []
	q2_set = []

	for u in xrange(-qm, qm):
		for v in xrange(-qm, qm):
			q = 4 * np.pi**2 * (u**2 / dim[0]**2 + v**2/dim[1]**2)
			q2 = u**2 * dim[1]/dim[0] + v**2 * dim[0]/dim[1]

			if q2 not in q2_set:
				q_set.append(q)
				q2_set.append(np.round(q2, 4))

	q_set = np.sqrt(np.sort(q_set, axis=None))
	q2_set = np.sort(q2_set, axis=None)

	return q_set, q2_set


def power_spectrum_coeff(coeff_2, qm, qu, dim):
	"""
	power_spectrum_coeff(coeff_2, qm, qu, dim)

	Returns power spectrum of average surface coefficients, corresponding to the frequencies in q2_set

	Parameters
	----------
	
	coeff_2:  float, array_like; shape=(n_waves**2)
		Square of optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------
	q_set:  float, array_like
		Set of frequencies for power spectrum histogram
	p_spec_hist:  float, array_like
		Power spectrum histogram of Fouier series coefficients
	
	"""

	q_set, q2_set = get_frequency_set(qm, dim)

	p_spec_hist = np.zeros(len(q2_set))
	p_spec_count = np.zeros(len(q2_set))

	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			set_index = np.round(u**2*dim[1]/dim[0] + v**2*dim[0]/dim[1], 4)

			if set_index != 0:
				p_spec = coeff_2[j] * ism.check_uv(u, v) / 4.
				p_spec_hist[q2_set == set_index] += p_spec
				p_spec_count[q2_set == set_index] += 1

	for i in xrange(len(q2_set)):
		if p_spec_count[i] != 0: p_spec_hist[i] *= 1. / p_spec_count[i]

	return q_set, p_spec_hist


def surface_tension_coeff(coeff_2, qm, qu, dim, T):
	"""
	surface_tension_coeff(coeff_2, qm, qu, dim, T)

	Returns spectrum of surface tension, corresponding to the frequencies in q2_set

	Parameters
	----------
	
	coeff_2:  float, array_like; shape=(n_waves**2)
		Square of optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	T:  float
		Average temperature of simulation (K)

	Returns
	-------
	q_set:  float, array_like
		Set of frequencies for power spectrum histogram
	gamma_hist:  float, array_like
		Surface tension histogram of Fouier series frequencies
	
	"""

	q_set, q2_set = get_frequency_set(qm, dim)

	gamma_hist = np.zeros(len(q2_set))
	gamma_count = np.zeros(len(q2_set))

	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			dot_prod = np.pi**2 * (u**2 * dim[1] / dim[0] + v**2 * dim[0] / dim[1])
			set_index = np.round(u**2*dim[1]/dim[0] + v**2*dim[0]/dim[1], 4)

			if set_index != 0:
				gamma = 1. / (ism.check_uv(u, v) * coeff_2[j] * 1E-20 * dot_prod)
				gamma_hist[q2_set == set_index] += gamma
				gamma_count[q2_set == set_index] += 1

	for i in xrange(len(q2_set)):
		if gamma_count[i] != 0: gamma_hist[i] *= con.k * 1E3 * T / gamma_count[i]

	return q_set, gamma_hist 


def cw_gamma_sr(q, gamma, kappa): return gamma + kappa * q**2


def cw_gamma_lr(q, gamma, kappa0, l0): return gamma + q**2 * (kappa0 + l0 * np.log(q))


def cw_gamma_dft(q, gamma, kappa, eta0, eta1): return gamma + eta0 * q + kappa * q**2 + eta1 * q**3

