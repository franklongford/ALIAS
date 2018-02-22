"""
*************** UTILITIES MODULE *******************

Generic functions to be used across multiple programs
within ALIAS.

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 22/2/2018 by Frank Longford
"""

import numpy as np
import subprocess, os, sys, pickle, tables

from scipy import stats
import scipy.constants as con

import mdtraj as md

SQRT2 = np.sqrt(2.)
SQRTPI = np.sqrt(np.pi)


def numpy_remove(list1, list2):
	"""
	numpy_remove(list1, list2)

	Deletes overlapping elements of list2 from list1
	"""

	return np.delete(list1, np.where(np.isin(list1, list2)))


def make_checkfile(checkfile_name):
	"""
	make_checkfile(checkfile_name)

	Creates checkfile for analysis, storing key paramtere
	"""

	checkfile = {}
	with file('{}.pkl'.format(checkfile_name), 'wb') as outfile:
		pickle.dump(checkfile, outfile, pickle.HIGHEST_PROTOCOL)

def read_checkfile(checkfile_name):
	"""
	read_checkfile(checkfile_name)

	Reads checkfile to lookup stored key paramters

	"""

	with file('{}.pkl'.format(checkfile_name), 'rb') as infile:
        	return pickle.load(infile)

def update_checkfile(checkfile_name, symb, obj):
	"""
	update_checkfile(checkfile_name, symb, obj)

	Updates checkfile parameter

	Parameters
	----------

	checkfile_name:  str
			Checkfile path + name
	symb:  str
			Key for checkfile dictionary of object obj
	obj:
			Parameter to be saved

	Returns
	-------

	checkfile:  dict
			Dictionary of key parameters

	"""

	with file('{}.pkl'.format(checkfile_name), 'rb') as infile:
        	checkfile = pickle.load(infile)
	checkfile[symb] = obj
	with file('{}.pkl'.format(checkfile_name), 'wb') as outfile:
        	pickle.dump(checkfile, outfile, pickle.HIGHEST_PROTOCOL)
	return checkfile


def get_sim_param(traj_dir, top_dir, traj_file, top_file):
	"""
	get_sim_param(traj_dir, top_dir, traj_file, top_file)

	Returns selected parameters of input trajectory and topology file using mdtraj
	
	Parameters
	----------

	top_dir:  str
			Directory of topology file
	traj_dir:  str
			Directory of trajectory file
	traj_file:  str
			Trajectory file name
	top_file:  str
			Topology file name

	Returns
	-------

	traj:  mdtraj obj
			Mdtraj trajectory object
	mol:  str, list
			List of residue types in simulation cell
	nframe:  int
			Number of frames sampled in traj_file
	dim:  float, array_like; shape=(3):
			Simulation cell dimensions (angstroms)
	"""

	traj = md.load('{}/{}'.format(traj_dir, traj_file), top='{}/{}'.format(top_dir, top_file))
	mol = list(set([molecule.name for molecule in traj.topology.residues]))
	nframe = int(traj.n_frames)
	dim = np.array(traj.unitcell_lengths[0]) * 10

	return traj, mol, nframe, dim


def molecules(xat, yat, zat, nmol, nsite, mol_M, mol_com):
	"""
	molecules(xat, yat, zat, nmol, nsite, mol_M, mol_com)

	Returns XYZ arrays of molecular positions"

	Parameters
	----------

	xat:  float, array_like; shape=(natom)
		Coordinates of atoms in x dimension
	yat:  float, array_like; shape=(natom)
		Coordinates of atoms in y dimension
	zat:  float, array_like; shape=(natom)
		Coordinates of atoms in z dimension
	nmol:  int
		Number of molecules in simulation
	nsite:  int
		Number of atomic sites per molecule
	mol_M:  float, array_like; shape=(natom)
		Masses of atomic sites in g mol-1
	mol_com:
		Mode of calculation: if 'COM', centre of mass is used, otherwise atomic site index is used

	Returns
	-------

	xmol:  float, array_like; shape=(nmol)
		Coordinates of molecules in x dimension
	ymol:  float, array_like; shape=(nmol)
		Coordinates of molecules in y dimension
	zmol:  float, array_like; shape=(nmol)
		Coordinates of molecules in z dimension

	"""
	if mol_com == 'COM':
		"USE CENTRE OF MASS AS MOLECULAR POSITION"
		xmol = np.sum(np.reshape(xat * mol_M, (nmol, nsite)), axis=1) 
		ymol = np.sum(np.reshape(yat * mol_M, (nmol, nsite)), axis=1)
		zmol = np.sum(np.reshape(zat * mol_M, (nmol, nsite)), axis=1)
	
	else:
		"USE SINGLE ATOM AS MOLECULAR POSITION"
		mol_list = np.arange(nmol) * nsite + int(mol_com)
		xmol = xat[mol_list]
		ymol = yat[mol_list]
		zmol = zat[mol_list]

	return xmol, ymol, zmol


def save_npy(directory, file_name, array):
	"""
	save_npy(directory, file_name, array)

	General purpose algorithm to save an array to a npy file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	array:  array_like (float);
		Data array to be saved
	"""

	with file('{}/{}.npy'.format(directory, file_name), 'w') as outfile:
		np.save(outfile, array)


def load_npy(directory, file_name, frames=[]):
	"""
	load_npy(directory, file_name, frames=[])

	General purpose algorithm to load an array from a npy file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	frames:  int, list (optional)
		Trajectory frames to load

	Returns
	-------

	array:  array_like (float);
		Data array to be loaded
	"""

	if len(frames) == 0: array = np.load('{}/{}.npy'.format(directory, file_name), mmap_mode='r')
	else: array = np.load('{}/{}.npy'.format(directory, file_name), mmap_mode='r')[frames]

	return array


def make_mol_com(traj, directory, file_name, natom, nmol, at_index, nframe, dim, nsite, M, mol_com):
	"""
	make_mol_com(traj, directory, file_name, natom, nmol, at_index, nframe, dim, nsite, M, mol_com)

	Generates molecular positions and centre of mass for each frame

	Parameters
	----------

	traj:  mdtraj obj
		Mdtraj trajectory object
	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	natom:  int
		Number of atoms in simulation
	nmol:  int
		Number of molecules in simulation
	at_index:  int, array_like; shape=(nsite*nmol)
		Indicies of atoms that are in molecules selected to determine intrinsic surface
	nframe:  int
		Number of frames in simulation trajectory
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	nsite:  int
		Number of atomic sites in molecule
	M:  float, array_like; shape=(nsite)
		Masses of atomic sites in molecule
	mol_com:
		Mode of calculation: if 'COM', centre of mass is used, otherwise atomic site index is used

	"""
	print "\n-----------CREATING POSITIONAL FILES------------\n"

	pos_dir = directory + '/pos'
	if not os.path.exists(pos_dir): os.mkdir(pos_dir)

	xmol = np.zeros((nframe, nmol))
	ymol = np.zeros((nframe, nmol))
	zmol = np.zeros((nframe, nmol))
	COM = np.zeros((nframe, 3))

	mol_M = np.array(M * nmol)
	mol_M /= mol_M.sum()

	XYZ = np.moveaxis(traj.xyz, 1, 2) * 10

	for frame in xrange(nframe):
		sys.stdout.write("PROCESSING {} out of {} IMAGES\r".format(frame, nframe) )
		sys.stdout.flush()

		COM[frame, :] = traj.xyz[frame].astype('float64').T.dot(mol_M) * 10

		xat = XYZ[frame][0][at_index]
		yat = XYZ[frame][1][at_index]
		zat = XYZ[frame][2][at_index]

		if nsite > 1: xmol[frame], ymol[frame], zmol[frame] = molecules(xat, yat, zat, nmol, nsite, mol_M, mol_com)
		else: xmol[frame], ymol[frame], zmol[frame] = xat, yat, zat
		
	file_name_pos = file_name + '_{}'.format(nframe)

	print '\nSAVING OUTPUT MOLECULAR POSITION FILES\n'
	save_npy(pos_dir, file_name_pos + '_xmol', xmol)
	save_npy(pos_dir, file_name_pos + '_ymol', ymol)
	save_npy(pos_dir, file_name_pos + '_zmol', zmol)
	save_npy(pos_dir, file_name_pos + '_com', COM)


def bubblesort(alist, key):
	"""
	bubblesort(alist, key)

	Sorts arrays 'alist' and 'key' by order of elements of 'key'
	"""

	for passnum in range(len(alist)-1,0,-1):
		for i in range(passnum):
			if key[i]>key[i+1]:
				temp = alist[i]
				alist[i] = alist[i+1]
				alist[i+1] = temp

				temp = key[i]
				key[i] = key[i+1]
				key[i+1] = temp

def unit_vector(vector, axis=-1):
	"""
	unit_vector(vector, axis=-1)

	Returns unit vector of vector
	"""

	vector = np.array(vector)
	magnitude_2 = np.sum(vector.T**2, axis=axis)
	u_vector = np.sqrt(vector**2 / magnitude_2) * np.sign(vector)

	return u_vector


def normalise(array):
	"""
	normalise(array)

	Returns an array normalised by the range of values in input array
	"""
	array = np.array(array)
	max_array = np.max(array)
	min_array = np.min(array)

	return (array - min_array) / (max_array - min_array)


def get_fourier_coeff(coeff, qm):
	"""
	get_fourier(coeff, nm)

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
        f_coeff = np.zeros(n_waves**2, dtype=complex)

        for u in xrange(-qm,qm+1):
                for v in xrange(-qm, qm+1):
                        index = n_waves * (u + qm) + (v + qm)

                        j1 = n_waves * (abs(u) + qm) + (abs(v) + qm)
                        j2 = n_waves * (-abs(u) + qm) + (abs(v) + qm)
                        j3 = n_waves * (abs(u) + qm) + (-abs(v) + qm)
                        j4 = n_waves * (-abs(u) + qm) + (-abs(v) + qm)

			if abs(u) + abs(v) == 0: f_coeff[index] = coeff[j1]

                        elif v == 0: f_coeff[index] = (coeff[j1] - np.sign(u) * 1j * coeff[j2]) / 2.
                        elif u == 0: f_coeff[index] = (coeff[j1] - np.sign(v) * 1j * coeff[j3]) / 2.

                        elif u < 0 and v < 0: f_coeff[index] = (coeff[j1] + 1j * (coeff[j2] + coeff[j3]) - coeff[j4]) / 4.
                        elif u > 0 and v > 0: f_coeff[index] = (coeff[j1] - 1j * (coeff[j2] + coeff[j3]) - coeff[j4]) / 4.

                        elif u < 0: f_coeff[index] = (coeff[j1] + 1j * (coeff[j2] - coeff[j3]) + coeff[j4]) / 4.
                        elif v < 0: f_coeff[index] = (coeff[j1] - 1j * (coeff[j2] - coeff[j3]) + coeff[j4]) / 4.

        return f_coeff


def linear(x, m, c): return m * x + c


def gaussian(x, mean, std): return np.exp(-(x-mean)**2 / (2 * std**2)) / (SQRT2 * std * SQRTPI)


def gaussian_convolution(arrays, centres, deltas, dim, nslice):
	"""
	gaussian_convolution(arrays, centres, deltas, dim, nslice)

	Convolution of distributions 'arrays' using a normal probability distribution with mean=centres and variance=deltas

	Parameters
	----------

	arrays:  float, array_like; shape=(n_arrays, n_dist)
		Set of arrays to convolute
	centres: float, array_like; shape=(n_arrays)	
		Set of mean values for normal probability distributions
	deltas: float, array_like; shape=(n_arrays)	
		Set of variances for normal probability distributions
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	nslice: int
		Number of bins in density histogram along axis normal to surface

	Returns
	-------

	conv_arrays:  float, array_like; shape=(n_arrays, n_dist)
		Set of convoluted arrays

	"""

	conv_arrays = np.zeros(arrays.shape)

	stds = np.sqrt(np.array(deltas))
	lslice = dim[2] / nslice

	Z1 = np.linspace(0, dim[2], nslice)
	max_std = np.max(stds)
	length = int(max_std / lslice) * 12
	ZG = np.arange(-lslice*length/2, lslice*length/2 + lslice/2, lslice)
	P_arrays = [[gaussian(z, 0, STD) for z in ZG ] for STD in stds]
	
	for n1, z1 in enumerate(Z1):
		for n2, z2 in enumerate(ZG):
			sys.stdout.write("PERFORMING GAUSSIAN SMOOTHING {0:.1%} COMPLETE \r".format(float(n1 * nslice + n2) / nslice**2) )
			sys.stdout.flush()

			indexes = [int((z1 - z2 - z0) / dim[2] * nslice) % nslice for z0 in centres]

			for i, array in enumerate(arrays):
				try: conv_arrays[i][n1] += array[indexes[i]] * P_arrays[i][n2] * lslice
				except IndexError: pass
	return conv_arrays


def make_earray(file_name, arrays, atom, sizes):
	"""
	make_earray(file_name, arrays, atom, sizes)

	General purpose algorithm to create an empty earray

	Parameters
	----------

	file_name:  str
		File name
	arrays:  str, list
		List of references for arrays in data table
	atom:  type
		Type of data in earray
	sizes:  int, tuple
		Shape of arrays in data set
	"""


	with tables.open_file(file_name, 'w') as outfile:
		for i, array in enumerate(arrays):
			outfile.create_earray(outfile.root, array, atom, sizes[i])


def make_hdf5(directory, file_name, shape, datatype):
	"""
	make_hdf5(directory, file_name, array, shape)

	General purpose algorithm to create an empty hdf5 file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	shape:  int, tuple
		Shape of dataset in hdf5 file
	datatype:  type
		Data type of dataset
	"""

	shape = (0,) + shape

	make_earray('{}/{}.hdf5'.format(directory, file_name), ['dataset'], datatype, [shape])


def load_hdf5(directory, file_name, frame='all'):
	"""
	load_hdf5(directory, file_name, frame='all')

	General purpose algorithm to load an array from a hdf5 file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	frame:  int (optional)
		Trajectory frame to load

	Returns
	-------

	array:  array_like (float);
		Data array to be loaded, same shape as object 'dataset' in hdf5 file
	"""

	with tables.open_file('{}/{}.hdf5'.format(directory, file_name), 'r') as infile:
		if frame == 'all': array = infile.root.dataset[:]
		else: array = infile.root.dataset[frame]

	return array


def save_hdf5(directory, file_name, array, frame, mode='a'):
	"""
	save_hdf5(directory, file_name, array, dataset, frame, mode='a')

	General purpose algorithm to save an array from a single frame a hdf5 file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	array:  array_like (float);
		Data array to be saved, must be same shape as object 'dataset' in hdf5 file
	frame:  int
		Trajectory frame to save
	mode:  str (optional)
		Option to append 'a' to hdf5 file or overwrite 'r+' existing data	
	"""

	if not mode: return

	shape = (1,) + array.shape

	with tables.open_file('{}/{}.hdf5'.format(directory, file_name), mode) as outfile:
		assert outfile.root.dataset.shape[1:] == shape[1:]
		if mode.lower() == 'a':
			write_array = np.zeros(shape)
			write_array[0] = array
			outfile.root.dataset.append(write_array)
		elif mode.lower() == 'r+':
			outfile.root.dataset[frame] = array


def shape_check_hdf5(directory, file_name):
	"""
	shape_check_hdf5(directory, file_name, nframe)

	General purpose algorithm to check the shape the dataset in a hdf5 file 

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed

	Returns
	-------

	shape_hdf5:  int, tuple
		Shape of object dataset in hdf5 file
	"""

	with tables.open_file('{}/{}.hdf5'.format(directory, file_name), 'r') as infile:
		shape_hdf5 = infile.root.dataset.shape

	return shape_hdf5




