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


def make_den_curve(directory, zmol, int_z_mol, int_dxdy_mol, coeff, nmol, nslice, nz, qm, dim):
	"""
	make_den_curve(directory, zmol, int_z_mol, int_dxdy_mol, coeff, nmol, nslice, nz, qm, dim)

	Creates density and curvature distributions normal to surface

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	zmol:  float, array_like; shape=(nmol)
		Molecular coordinates in z dimension
	int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
		Molecular distances from intrinsic surface
	int_dxdy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
		First derivatives of intrinsic surface wrt x and y at xmol, ymol
	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	nmol:  int
		Number of molecules in simulation
	nslice: int
		Number of bins in density histogram along axis normal to surface
	nz: int (optional)
		Number of bins in curvature histogram along axis normal to surface (default=100)
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	count_corr_array:  int, array_like; shape=(qm+1, nslice, nz)
		Number histogram binned by molecular position along z axis and mean curvature H across qm resolutions 

	"""

	lslice = dim[2] / nslice

	count_corr_array = np.zeros((qm+1, nslice, nz))

	for qu in xrange(qm+1):

		temp_count_corr_array = np.zeros((nslice, nz))

		int_z1 = int_z_mol[0][qu]
		int_z2 = int_z_mol[1][qu]

		z1 = zmol - int_z1
		z2 = -zmol + int_z2

		dzx1 = int_dxdy_mol[0][qu]
		dzy1 = int_dxdy_mol[1][qu]
		dzx2 = int_dxdy_mol[2][qu]
		dzy2 = int_dxdy_mol[3][qu]

		index1_mol = np.array((z1 + dim[2]/2.) * nslice / dim[2], dtype=int) % nslice
		index2_mol = np.array((z2 + dim[2]/2.) * nslice / dim[2], dtype=int) % nslice

		normal1 = ut.unit_vector(np.array([-dzx1, -dzy1, np.ones(nmol)]))
		normal2 = ut.unit_vector(np.array([-dzx2, -dzy2, np.ones(nmol)]))

		index1_nz = np.array(abs(normal1[2]) * nz, dtype=int) % nz
		index2_nz = np.array(abs(normal2[2]) * nz, dtype=int) % nz

		temp_count_corr_array += np.histogram2d(index1_mol, index1_nz, bins=[nslice, nz], range=[[0, nslice], [0, nz]])[0]
		temp_count_corr_array += np.histogram2d(index2_mol, index2_nz, bins=[nslice, nz], range=[[0, nslice], [0, nz]])[0]

		count_corr_array[qu] += temp_count_corr_array

	return count_corr_array


def create_intrinsic_den_curve_dist(directory, file_name, qm, n0, phi, nframe, nslice, dim, nz=100, recon=False, ow_hist=False):
	"""
	create_intrinsic_den_curve_dist(directory, file_name, qm, n0, phi, nframe, nslice, dim, nz=100, nnz=100, recon=False, ow_count=False)

	Calculate density and curvature distributions across surface

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed.
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	n0:  int
		Maximum number of molecular pivots in intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	nframe:  int
		Number of frames in simulation trajectory
	nslice: int
		Number of bins in density histogram along axis normal to surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	nz: int (optional)
		Number of bins in curvature histogram along axis normal to surface (default=100)
	recon:  bool (optional)
		Whether to use surface reconstructe coefficients (default=False)
	ow_count:  bool (optional)
		Whether to overwrite density and curvature distributions (default=False)
	"""

	print"\n--- Running Intrinsic Density and Curvature Routine --- \n"

	surf_dir = directory + 'surface/'
	pos_dir = directory + 'pos/'
	intpos_dir = directory + 'intpos/'
	intden_dir = directory + 'intden/'
	if not os.path.exists(intden_dir): os.mkdir(intden_dir)

	lslice = dim[2] / nslice

	file_name_pos = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
	file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
	file_name_hist = '{}_{}_{}_{}_{}_{}_{}'.format(file_name, nslice, nz, qm, n0, int(1./phi + 0.5), nframe)	

	if recon:
		file_name_pos += '_R'
		file_name_hist += '_R'
		file_name_coeff += '_R'

	if not os.path.exists(intden_dir + file_name_hist + '_count_corr.hdf5'):
		ut.make_hdf5(intden_dir + file_name_hist + '_count_corr', (qm+1, nslice, nz), tables.Float64Atom())
		file_check = False

	elif not ow_hist:
		"Checking number of frames in current distribution files"
		try: file_check = (ut.shape_check_hdf5(intden_dir + file_name_hist + '_count_corr') == (nframe, qm+1, nslice, nz))
		except: file_check = False
	else:file_check = False

	if not file_check:
		zmol = ut.load_npy(pos_dir + file_name + '_{}_zmol'.format(nframe))
		COM = ut.load_npy(pos_dir + file_name + '_{}_com'.format(nframe))
		nmol = zmol.shape[1]
		com_tile = np.moveaxis(np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
		zmol = zmol - com_tile

		for frame in xrange(nframe):

			"Checking number of frames in hdf5 files"
			frame_check_count_corr = (ut.shape_check_hdf5(intden_dir + file_name_hist + '_count_corr') <= frame)

			if frame_check_count_corr: mode_count_corr = 'a'
			elif ow_hist: mode_count_corr = 'r+'
			else: mode_count_corr = False

			if not mode_count_corr:pass
			else:
				sys.stdout.write("Calculating position and curvature distributions: frame {}\r".format(frame))
				sys.stdout.flush()

				coeff = ut.load_hdf5(surf_dir + file_name_coeff + '_coeff', frame)
				int_z_mol = ut.load_hdf5(intpos_dir + file_name_pos + '_int_z_mol', frame)
				int_dxdy_mol = ut.load_hdf5(intpos_dir + file_name_pos + '_int_dxdy_mol', frame)

				count_corr_array = make_den_curve(directory, zmol[frame], int_z_mol, int_dxdy_mol, coeff, nmol, nslice, nz, qm, dim)
				ut.save_hdf5(intden_dir + file_name_hist + '_count_corr', count_corr_array, frame, mode_count_corr)


def av_intrinsic_distributions(directory, file_name, dim, nslice, qm, n0, phi, nframe, nsample, nz=100, recon=False, ow_dist=False):
	"""
	av_intrinsic_distributions(directory, file_name, dim, nslice, qm, n0, phi, nframe, nsample, nz=100, recon=False, ow_dist=False)

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

	file_name_hist = '{}_{}_{}_{}_{}_{}_{}'.format(file_name, nslice, nz, qm, n0, int(1./phi + 0.5), nframe)
	if recon: file_name_hist += '_R'

	if not os.path.exists(intden_dir + file_name_hist + '_int_den_curve.npy'):

		int_den_curve_matrix = np.zeros((qm+1, nslice, nz))

		print "\n--- Loading in Density and Curvature Distributions ---\n"

		lslice = dim[2] / nslice
		Vslice = dim[0] * dim[1] * lslice

		for frame in xrange(nsample):
			sys.stdout.write("Frame {}\r".format(frame))
			sys.stdout.flush()

			count_corr_array = ut.load_hdf5(intden_dir + file_name_hist + '_count_corr', frame)
			int_den_curve_matrix += count_corr_array / (nsample * Vslice)

		ut.save_npy(intden_dir + file_name_hist + '_int_den_curve', int_den_curve_matrix)

	else:
		int_den_curve_matrix = ut.load_npy(intden_dir + file_name_hist + '_int_den_curve')

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


def coeff_slice(coeff, qm, qu):

	n_waves_qm = 2 * qm + 1
	n_waves_qu = 2 * qu + 1

	index_1 = qm - qu
	index_2 = index_1 + n_waves_qu

	coeff_matrix = np.reshape(coeff, (n_waves_qm, n_waves_qm))
	coeff_qu = coeff_matrix[[slice(index_1, index_2) for _ in coeff_matrix.shape]].flatten()

	return coeff_qu
	

def auv2_to_f2(auv2, qm):

	f2 = np.zeros((2*qm+1)**2)

	for u in xrange(-qm, qm+1):
		for v in xrange(-qm, qm+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			f2[j] = auv2[j] * ut.check_uv(u, v) / 4.

	return f2


def auv_xy_correlation(auv_2, qm, qu):

	auv_2[len(auv_2)/2] = 0
	f2 = auv2_to_f2(auv_2, qm)

	f2_qm = auv_qm(f2, qm, qu).reshape(((2*qu+1), (2*qu+1)))
	xy_corr = np.fft.fftshift(np.fft.ifftn(f2_qm))
	#xy_corr = np.fft.ifftn(f2_qm)

	return np.abs(xy_corr) * (2*qu+1)**2 / np.sum(f2_qm)

