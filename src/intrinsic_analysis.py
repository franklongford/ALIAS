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
from intrinsic_sampling_method import xi, wave_function, d_wave_function, dd_wave_function, check_uv

import os, sys, time, tables


vcheck = np.vectorize(check_uv)

def make_pos_dxdy(xmol, ymol, coeff, nmol, dim, qm):
	"""
	make_pos_dxdy(xmol, ymol, coeff, nmol, dim, qm)

	Calculate distances and derivatives at each molecular position with respect to intrinsic surface

	Parameters
	----------

	xmol:  float, array_like; shape=(nmol)
		Molecular coordinates in x dimension
	ymol:  float, array_like; shape=(nmol)
		Molecular coordinates in y dimension
	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	nmol:  int
		Number of molecules in simulation
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface

	Returns
	-------

	int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
		Molecular distances from intrinsic surface
	int_dxdy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
		First derivatives of intrinsic surface wrt x and y at xmol, ymol
	int_ddxddy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
		Second derivatives of intrinsic surface wrt x and y at xmol, ymol 

	"""
	
	int_z_mol = np.zeros((qm+1, 2, nmol))
	int_dxdy_mol = np.zeros((qm+1, 4, nmol)) 
	int_ddxddy_mol = np.zeros((qm+1, 4, nmol))

	tmp_int_z_mol = np.zeros((2, nmol))
	tmp_dxdy_mol = np.zeros((4, nmol)) 
	tmp_ddxddy_mol = np.zeros((4, nmol))
	
	for qu in xrange(qm+1):

		if qu == 0:
			j = (2 * qm + 1) * qm + qm
			f_x = wave_function(xmol, 0, dim[0])
			f_y = wave_function(ymol, 0, dim[1])

			tmp_int_z_mol[0] += f_x * f_y * coeff[0][j]
			tmp_int_z_mol[1] += f_x * f_y * coeff[1][j]

		else:
			for u in [-qu, qu]:
				for v in xrange(-qu, qu+1):
					j = (2 * qm + 1) * (u + qm) + (v + qm)

					f_x = wave_function(xmol, u, dim[0])
					f_y = wave_function(ymol, v, dim[1])
					df_dx = d_wave_function(xmol, u, dim[0])
					df_dy = d_wave_function(ymol, v, dim[1])
					ddf_ddx = dd_wave_function(xmol, u, dim[0])
					ddf_ddy = dd_wave_function(ymol, v, dim[1])

					tmp_int_z_mol[0] += f_x * f_y * coeff[0][j]
					tmp_int_z_mol[1] += f_x * f_y * coeff[1][j]
					tmp_dxdy_mol[0] += df_dx * f_y * coeff[0][j]
					tmp_dxdy_mol[1] += f_x * df_dy * coeff[0][j]
					tmp_dxdy_mol[2] += df_dx * f_y * coeff[1][j]
					tmp_dxdy_mol[3] += f_x * df_dy * coeff[1][j]
					tmp_ddxddy_mol[0] += ddf_ddx * f_y * coeff[0][j]
					tmp_ddxddy_mol[1] += f_x * ddf_ddy * coeff[0][j]
					tmp_ddxddy_mol[2] += ddf_ddx * f_y * coeff[1][j]
					tmp_ddxddy_mol[3] += f_x * ddf_ddy * coeff[1][j]

			for u in xrange(-qu+1, qu):
				for v in [-qu, qu]:
					j = (2 * qm + 1) * (u + qm) + (v + qm)

					f_x = wave_function(xmol, u, dim[0])
					f_y = wave_function(ymol, v, dim[1])
					df_dx = d_wave_function(xmol, u, dim[0])
					df_dy = d_wave_function(ymol, v, dim[1])
					ddf_ddx = dd_wave_function(xmol, u, dim[0])
					ddf_ddy = dd_wave_function(ymol, v, dim[1])

					tmp_int_z_mol[0] += f_x * f_y * coeff[0][j]
					tmp_int_z_mol[1] += f_x * f_y * coeff[1][j]
					tmp_dxdy_mol[0] += df_dx * f_y * coeff[0][j]
					tmp_dxdy_mol[1] += f_x * df_dy * coeff[0][j]
					tmp_dxdy_mol[2] += df_dx * f_y * coeff[1][j]
					tmp_dxdy_mol[3] += f_x * df_dy * coeff[1][j]
					tmp_ddxddy_mol[0] += ddf_ddx * f_y * coeff[0][j]
					tmp_ddxddy_mol[1] += f_x * ddf_ddy * coeff[0][j]
					tmp_ddxddy_mol[2] += ddf_ddx * f_y * coeff[1][j]
					tmp_ddxddy_mol[3] += f_x * ddf_ddy * coeff[1][j]

		int_z_mol[qu] += tmp_int_z_mol
		int_dxdy_mol[qu] += tmp_dxdy_mol
		int_ddxddy_mol[qu] += tmp_ddxddy_mol

	int_z_mol = np.swapaxes(int_z_mol, 0, 1)
	int_dxdy_mol = np.swapaxes(int_dxdy_mol, 0, 1)
	int_ddxddy_mol = np.swapaxes(int_ddxddy_mol, 0, 1)
	
	return int_z_mol, int_dxdy_mol, int_ddxddy_mol


def create_intrinsic_positions_dxdyz(directory, file_name, nmol, nframe, qm, n0, phi, dim, recon=0, ow_pos=False):
	"""
	create_intrinsic_positions_dxdyz(directory, file_name, nmol, nframe, qm, n0, phi, dim, recon, ow_pos)

	Calculate distances and derivatives at each molecular position with respect to intrinsic surface in simulation frame

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed.
	nmol:  int
		Number of molecules in simulation
	nframe:  int
		Number of frames in simulation trajectory
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	n0:  int
		Maximum number of molecular pivots in intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	recon:  bool (default=False)
		Whether to use surface reconstructe coefficients
	ow_pos:  bool (default=False)
		Whether to overwrite positions and derivatives (default=False)

	"""

	print"\n--- Running Intrinsic Positions and Derivatives Routine ---\n"

	n_waves = 2 * qm + 1
	
	surf_dir = directory + 'surface/'
	pos_dir = directory + 'pos/'
	intpos_dir = directory + 'intpos/'
	if not os.path.exists(intpos_dir): os.mkdir(intpos_dir)

	file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	file_name_pos = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)

	if recon: 
		file_name_coeff += '_r'
		file_name_pos += '_r'

	if not os.path.exists('{}/{}_int_z_mol.hdf5'.format(intpos_dir, file_name_pos)):
		ut.make_hdf5(intpos_dir + file_name_pos + '_int_z_mol', (2, qm+1, nmol), tables.Float64Atom())
		ut.make_hdf5(intpos_dir + file_name_pos + '_int_dxdy_mol', (4, qm+1, nmol), tables.Float64Atom())
		ut.make_hdf5(intpos_dir + file_name_pos + '_int_ddxddy_mol', (4, qm+1, nmol), tables.Float64Atom())
		file_check = False

	elif not ow_pos:
		"Checking number of frames in current distance files"
		try:
			file_check = (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_z_mol') == (nframe, 2, qm+1, nmol))
			file_check *= (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_dxdy_mol') == (nframe, 4, qm+1, nmol))
			file_check *= (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_ddxddy_mol') == (nframe, 4, qm+1, nmol))
		except: file_check = False
	else: file_check = False

	if not file_check:
		xmol = ut.load_npy(pos_dir + file_name + '_{}_xmol'.format(nframe), frames=range(nframe))
		ymol = ut.load_npy(pos_dir + file_name + '_{}_ymol'.format(nframe), frames=range(nframe))

		for frame in xrange(nframe):

			"Checking number of frames in int_z_mol file"
			frame_check_int_z_mol = (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_z_mol')[0] <= frame)
			frame_check_int_dxdy_mol = (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_dxdy_mol')[0] <= frame)
			frame_check_int_ddxddy_mol = (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_ddxddy_mol')[0] <= frame)

			if frame_check_int_z_mol: mode_int_z_mol = 'a'
			elif ow_pos: mode_int_z_mol = 'r+'
			else: mode_int_z_mol = False

			if frame_check_int_dxdy_mol: mode_int_dxdy_mol = 'a'
			elif ow_pos: mode_int_dxdy_mol = 'r+'
			else: mode_int_dxdy_mol = False

			if frame_check_int_ddxddy_mol: mode_int_ddxddy_mol = 'a'
			elif ow_pos: mode_int_ddxddy_mol = 'r+'
			else: mode_int_ddxddy_mol = False

			if not mode_int_z_mol and not mode_int_dxdy_mol and not mode_int_ddxddy_mol: pass
			else:
				sys.stdout.write("Calculating molecular distances and derivatives: frame {}\r".format(frame))
				sys.stdout.flush()
			
				coeff = ut.load_hdf5(surf_dir + file_name_coeff + '_coeff', frame)

				int_z_mol, int_dxdy_mol, int_ddxddy_mol = make_pos_dxdy(xmol[frame], ymol[frame], coeff, nmol, dim, qm)
				ut.save_hdf5(intpos_dir + file_name_pos + '_int_z_mol', int_z_mol, frame, mode_int_z_mol)
				ut.save_hdf5(intpos_dir + file_name_pos + '_int_dxdy_mol', int_dxdy_mol, frame, mode_int_dxdy_mol)
				ut.save_hdf5(intpos_dir + file_name_pos + '_int_ddxddy_mol', int_ddxddy_mol, frame, mode_int_ddxddy_mol)


def make_int_mol_count(zmol, int_z_mol, nmol, nslice, qm, dim):
	"""
	make_int_mol_count(zmol, int_z_mol, nmol, nslice, qm, dim)

	Creates density histogram

	Parameters
	----------

	zmol:  float, array_like; shape=(nmol)
		Molecular coordinates in z dimension
	int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
		Molecular distances from intrinsic surface
	nmol:  int
		Number of molecules in simulation
	nslice: int
		Number of bins in density histogram along axis normal to surface
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	mol_count_array:  int, array_like; shape=(qm+1, nslice, nz)
		Number histogram binned by molecular position along z axis and mean curvature H across qm resolutions 

	"""

	lslice = dim[2] / nslice
	mol_count_array = np.zeros((qm+1, nslice))

	for qu in xrange(qm+1):

		temp_mol_count_array = np.zeros((nslice))

		int_z1 = int_z_mol[0][qu]
		int_z2 = int_z_mol[1][qu]

		z1 = zmol - int_z1 + dim[2]
		z2 = -(zmol - int_z2) + dim[2]

		z1 -= dim[2] * np.array(z1 / dim[2], dtype=int)
		z2 -= dim[2] * np.array(z2 / dim[2], dtype=int)

		temp_mol_count_array += np.histogram(z1, bins=nslice, range=[0, dim[2]])[0]
		temp_mol_count_array += np.histogram(z2, bins=nslice, range=[0, dim[2]])[0]

		mol_count_array[qu] += temp_mol_count_array

	return mol_count_array


def den_curve_hist(zmol, int_z_mol, int_ddxddy_mol, nmol, nslice, nz, qm, dim, max_H=12):
	"""
	den_curve_hist(directory, zmol, int_z_mol, int_ddxddy_mol, nmol, nslice, nz, qm, dim)

	Creates density and mean curvature histograms

	Parameters
	----------

	zmol:  float, array_like; shape=(nmol)
		Molecular coordinates in z dimension
	int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
		Molecular distances from intrinsic surface
	int_ddxddy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
		Second derivatives of intrinsic surface wrt x and y at xmol, ymol
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

	import matplotlib.pyplot as plt

	for qu in xrange(qm+1):

		temp_count_corr_array = np.zeros((nslice, nz))

		int_z1 = int_z_mol[0][qu]
		int_z2 = int_z_mol[1][qu]

		z1 = zmol - int_z1
		z2 = zmol - int_z2

		z1 -= dim[2] * np.array(2 * z1 / dim[2], dtype=int)
		z2 -= dim[2] * np.array(2 * z2 / dim[2], dtype=int)

		#plt.hist(z1, bins=100)
		#plt.hist(z2, bins=100)
		#plt.show()

		#dzx1 = int_dxdy_mol[0][qu]
		#dzy1 = int_dxdy_mol[1][qu]
		#dzx2 = int_dxdy_mol[2][qu]
		#dzy2 = int_dxdy_mol[3][qu]

		ddzx1 = int_ddxddy_mol[0][qu]
		ddzy1 = int_ddxddy_mol[1][qu]
		ddzx2 = int_ddxddy_mol[2][qu]
		ddzy2 = int_ddxddy_mol[3][qu]

		#index1_mol = np.array((z1 + dim[2]/2.) * nslice / dim[2], dtype=int) % nslice
		#index2_mol = np.array((z2 + dim[2]/2.) * nslice / dim[2], dtype=int) % nslice

		#normal1 = ut.unit_vector(np.array([-dzx1, -dzy1, np.ones(nmol)]))
		#normal2 = ut.unit_vector(np.array([-dzx2, -dzy2, np.ones(nmol)]))

		#index1_nz = np.array(abs(normal1[2]) * nz, dtype=int) % nz
		#index2_nz = np.array(abs(normal2[2]) * nz, dtype=int) % nz

		H1 = abs(ddzx1 + ddzy1)
		H2 = abs(ddzx2 + ddzy2)

		#index1_H = np.array(H1 * nz, dtype=int) % nz
		#index2_H = np.array(H2 * nz, dtype=int) % nz

		temp_count_corr_array += np.histogram2d(z1, H1, bins=[nslice, nz], range=[[-dim[2]/2, dim[2]/2], [0, max_H]])[0]
		temp_count_corr_array += (np.histogram2d(z2, H2, bins=[nslice, nz], range=[[-dim[2]/2, dim[2]/2], [0, max_H]])[0])[::-1]

		count_corr_array[qu] += temp_count_corr_array

	return count_corr_array


def create_intrinsic_den_curve_hist(directory, file_name, qm, n0, phi, nframe, nslice, dim, nz=100, recon=0, ow_hist=False):
	"""
	create_intrinsic_den_curve_hist(directory, file_name, qm, n0, phi, nframe, nslice, dim, nz=100, recon=False, ow_hist=False)

	Calculate density and curvature histograms across surface

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
	file_name_hist = '{}_{}_{}_{}_{}_{}_{}'.format(file_name, nslice, nz, qm, n0, int(1./phi + 0.5), nframe)	

	if recon:
		file_name_pos += '_r'
		file_name_hist += '_r'

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
			frame_check_count_corr = (ut.shape_check_hdf5(intden_dir + file_name_hist + '_count_corr')[0] <= frame)

			if frame_check_count_corr: mode_count_corr = 'a'
			elif ow_hist: mode_count_corr = 'r+'
			else: mode_count_corr = False
			
			if not mode_count_corr:pass
			else:
				sys.stdout.write("Calculating position and curvature distributions: frame {}\r".format(frame))
				sys.stdout.flush()

				int_z_mol = ut.load_hdf5(intpos_dir + file_name_pos + '_int_z_mol', frame)
				int_ddxddy_mol = ut.load_hdf5(intpos_dir + file_name_pos + '_int_ddxddy_mol', frame)

				count_corr_array = den_curve_hist(zmol[frame], int_z_mol, int_ddxddy_mol, nmol, nslice, nz, qm, dim)
				ut.save_hdf5(intden_dir + file_name_hist + '_count_corr', count_corr_array, frame, mode_count_corr)
				

def av_intrinsic_distributions(directory, file_name, dim, nslice, qm, n0, phi, nframe, nsample, nz=100, recon=0, ow_dist=False):
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
	file_name_dist = '{}_{}_{}_{}_{}_{}_{}'.format(file_name, nslice, nz, qm, n0, int(1./phi + 0.5), nsample)

	if recon: 
		file_name_hist += '_r'
		file_name_dist += '_r'

	if not os.path.exists(intden_dir + file_name_dist + '_int_den_curve.npy') or ow_dist:

		int_den_curve_matrix = np.zeros((qm+1, nslice, nz))

		print "\n--- Loading in Density and Curvature Distributions ---\n"

		lslice = dim[2] / nslice
		Vslice = dim[0] * dim[1] * lslice

		for frame in xrange(nsample):
			sys.stdout.write("Frame {}\r".format(frame))
			sys.stdout.flush()

			count_corr_array = ut.load_hdf5(intden_dir + file_name_hist + '_count_corr', frame)
			int_den_curve_matrix += count_corr_array / (nsample * Vslice)

		ut.save_npy(intden_dir + file_name_dist + '_int_den_curve', int_den_curve_matrix)

	else:
		int_den_curve_matrix = ut.load_npy(intden_dir + file_name_dist + '_int_den_curve')

	int_density = np.sum(int_den_curve_matrix, axis=2) / 2.
	int_curvature = np.sum(np.moveaxis(int_den_curve_matrix, 1, 2), axis=2) / 2.

	return int_den_curve_matrix, int_density, int_curvature


def xi_var(coeff, qm, qu, dim):
	"""
	xi_var(coeff, qm, qu, dim)

	Calculate average variance of surface heights

	Parameters
	----------

	coeff:	float, array_like; shape=(n_frame, n_waves**2)
		Optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	calc_var: float
		Variance of surface heights across whole surface

	"""

	nframe = coeff.shape[0]
	n_waves = 2 * qm +1
	nxy = 40
	
	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
	wave_filter = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
	indices = np.argwhere(wave_filter).flatten()
	Psi = vcheck(u_array, v_array)[indices] / 4.

	coeff_filter = coeff[:,:,indices]
	mid_point = len(indices) / 2 

	av_coeff = np.mean(coeff_filter[:, :,mid_point], axis=0)
	av_coeff_2 = np.mean(coeff_filter**2, axis=(0, 1)) * Psi
	
	calc_var = np.sum(av_coeff_2) - np.mean(av_coeff**2, axis=0)

	return calc_var


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

	if np.isscalar(x) and np.isscalar(y):
		u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
		v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
		wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
		indices = np.argwhere(wave_check).flatten()

		fuv = wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
		H = -4 * np.pi**2 * np.sum((u_array[indices]**2 / dim[0]**2 + v_array[indices]**2 / dim[1]**2) * fuv * coeff[indices])
	else:
		H_array = np.zeros(x.shape)
		for u in xrange(-qu, qu+1):
			for v in xrange(-qu, qu+1):
				j = (2 * qm + 1) * (u + qm) + (v + qm)
				H_array += wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * (u**2 / dim[0]**2 + v**2 / dim[1]**2) * coeff[j]
		H = -4 * np.pi**2 * H_array

	return H


def H_var_coeff(coeff, qm, qu, dim):
	"""
	H_var_coeff(coeff, qm, qu, dim)

	Variance of mean curvature H across surface determined by coeff at resolution qu

	Parameters
	----------

	coeff:	float, array_like; shape=(n_frame, n_waves**2)
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
		Variance of mean curvature H across whole surface

	"""

	if qu == 0: return 0


	nframe = coeff.shape[0]
	n_waves = 2 * qm +1
	
	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
	wave_filter = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
	indices = np.argwhere(wave_filter).flatten()
	Psi = vcheck(u_array, v_array)[indices] / 4.

	coeff_filter = coeff[:,:,indices]
	av_coeff_2 = np.mean(coeff_filter**2, axis=(0, 1)) * Psi

	H_var_array = vcheck(u_array[indices], v_array[indices]) * av_coeff_2[indices]
	H_var_array *= (u_array[indices]**4 / dim[0]**4 + v_array[indices]**4 / dim[1]**4 + 2 * u_array[indices]**2 * v_array[indices]**2 / (dim[0]**2 * dim[1]**2))
	H_var = 16 * np.pi**4 * np.sum(H_var_array) 

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

	Returns set of unique frequencies in Fourier series

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
				p_spec = coeff_2[j] * check_uv(u, v) / 4.
				p_spec_hist[q2_set == set_index] += p_spec
				p_spec_count[q2_set == set_index] += 1

	for i in xrange(len(q2_set)):
		if p_spec_count[i] != 0: p_spec_hist[i] *= 1. / p_spec_count[i]

	return q_set, p_spec_hist


def surface_tension_coeff(coeff_2, qm, qu, dim, T, error=False, std=None):
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
				if error: 
					gamma = con.k * T * 1E23 / (check_uv(u, v) * coeff_2[j]**2 * dot_prod)
					gamma_hist[q2_set == set_index] += (gamma * std[j])#**2
				else: 
					gamma = con.k * T * 1E23 / (check_uv(u, v) * coeff_2[j] * dot_prod)
					gamma_hist[q2_set == set_index] += gamma
				gamma_count[q2_set == set_index] += 1

	#if error: gamma_hist = np.sqrt(gamma_hist)

	for i in xrange(len(q2_set)):
		if gamma_count[i] != 0: gamma_hist[i] /= gamma_count[i]
	
	return q_set, gamma_hist 


def cw_gamma_sr(q, gamma, kappa): return gamma + kappa * q**2


def cw_gamma_lr(q, gamma, kappa0, l0): return gamma + q**2 * (kappa0 + l0 * np.log(q))


def coeff_slice(coeff, qm, qu):
	"""
	coeff_slice(coeff, qm, qu)

	Truncates coeff array up to qu resolution
	"""

	n_waves_qm = 2 * qm + 1
	n_waves_qu = 2 * qu + 1

	index_1 = qm - qu
	index_2 = index_1 + n_waves_qu

	coeff_matrix = np.reshape(coeff, (n_waves_qm, n_waves_qm))
	coeff_qu = coeff_matrix[[slice(index_1, index_2) for _ in coeff_matrix.shape]].flatten()

	return coeff_qu
	

def coeff_to_fourier(coeff, qm, dim):
	"""
	coeff_to_fourier(coeff, nm)

	Returns Fouier coefficients for Fouier series representing intrinsic surface from linear algebra coefficients

	Parameters
	----------

	coeff:	float, array_like; shape=(n_waves**2)
		Optimised linear algebra surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	
	Returns
	-------

	f_coeff:  float, array_like; shape=(n_waves**2)
		Optimised Fouier surface coefficients
	
	"""

	n_waves = 2 * qm + 1

	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

	frequencies = np.pi * 2 * (u_array.reshape(n_waves, n_waves) / dim[0] + v_array.reshape(n_waves, n_waves) / dim[1])

        amplitudes = np.zeros(n_waves**2, dtype=complex)

        for u in xrange(-qm,qm+1):
                for v in xrange(-qm, qm+1):
                        index = n_waves * (u + qm) + (v + qm)

                        j1 = n_waves * (abs(u) + qm) + (abs(v) + qm)
                        j2 = n_waves * (-abs(u) + qm) + (abs(v) + qm)
                        j3 = n_waves * (abs(u) + qm) + (-abs(v) + qm)
                        j4 = n_waves * (-abs(u) + qm) + (-abs(v) + qm)

			if abs(u) + abs(v) == 0: amplitudes[index] = coeff[j1]

                        elif v == 0: amplitudes[index] = (coeff[j1] - np.sign(u) * 1j * coeff[j2]) / 2.
                        elif u == 0: amplitudes[index] = (coeff[j1] - np.sign(v) * 1j * coeff[j3]) / 2.

                        elif u < 0 and v < 0: amplitudes[index] = (coeff[j1] + 1j * (coeff[j2] + coeff[j3]) - coeff[j4]) / 4.
                        elif u > 0 and v > 0: amplitudes[index] = (coeff[j1] - 1j * (coeff[j2] + coeff[j3]) - coeff[j4]) / 4.

                        elif u < 0: amplitudes[index] = (coeff[j1] + 1j * (coeff[j2] - coeff[j3]) + coeff[j4]) / 4.
                        elif v < 0: amplitudes[index] = (coeff[j1] - 1j * (coeff[j2] - coeff[j3]) + coeff[j4]) / 4.

        return amplitudes, frequencies 


def coeff_to_fourier_2(coeff_2, qm, dim):
	"""
	coeff_to_fouier_2(coeff_2, qm)

	Converts square coefficients to square fouier coefficients
	"""

	n_waves = 2 * qm + 1
	
	i_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int)
	j_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int)

	u_mat, v_mat = np.meshgrid(np.arange(-qm, qm+1), 
				   np.arange(-qm, qm+1))
	x_mat, y_mat = np.meshgrid(np.linspace(0, 1 / dim[0], n_waves), 
				   np.linspace(0, 1 / dim[1], n_waves))

	print(x_mat, y_mat)

	Psi = vcheck(u_mat.flatten(), v_mat.flatten()) / 4.
	frequencies = np.pi * 2 * (u_mat * x_mat + y_mat * v_mat) / n_waves
	amplitudes_2 = np.reshape(Psi * coeff_2, (n_waves, n_waves))

	A = np.zeros((n_waves, n_waves))

	for i in xrange(n_waves):
		for j in xrange(n_waves):
			A[i][j] += (amplitudes_2 * np.exp(-2 * np.pi * 1j * (u_mat * x_mat[i][j] + y_mat[i][j] * v_mat) / n_waves)).sum()
	

	return A, frequencies 


def xy_correlation(coeff_2, qm, qu, dim):
	"""
	xy_correlation(coeff_2, qm, qu, dim)

	Return correlation across xy plane using Wiener-Khinchin theorem

	Parameters
	----------

	coeff_2:  float, array_like; shape=(n_waves**2)
		Square of optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface

	Returns
	-------

	xy_corr:  float, array_like; shape=(n_waves_qu**2)
		Length correlation function across xy plane

	"""

	coeff_2[len(coeff_2)/2] = 0
	coeff_2_slice = coeff_slice(coeff_2, qm, qu)

	xy_corr, frequencies = coeff_to_fourier_2(coeff_2_slice, qu, dim)
	#xy_corr = np.abs(amplitudes_2) / np.mean(amplitudes_2)

	return xy_corr, frequencies


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


def create_intrinsic_den_curve_dist(directory, file_name, qm, n0, phi, nframe, nslice, dim, nz=100, recon=0, ow_hist=False):
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
		file_name_pos += '_r'
		file_name_hist += '_r'
		file_name_coeff += '_r'

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

