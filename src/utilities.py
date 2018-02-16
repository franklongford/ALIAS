"""
*************** UTILITIES MODULE *******************

Generic functions to be used across multiple programs
within ALIAS.

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford, Sam Munday

Last modified 14/12/18 by Frank Longford
"""

import numpy as np
import subprocess, os, sys, pickle, tables

from scipy import stats
import scipy.constants as con

import mdtraj as md

import matplotlib.pyplot as plt

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


def make_mol_com(traj, directory, file_name, natom, nmol, at_index, nframe, dim, nsite, M, mol, mol_com):
	"""
	make_mol_com(traj, directory, file_name, natom, nmol, at_index, nframe, dim, nsite, M, mol, mol_com)

	Generates molecular positions and centre of mass for each frame
	"""
	print "\n-----------CREATING POSITIONAL FILES------------\n"

	if not os.path.exists("{}/pos".format(directory)): os.mkdir("{}/pos".format(directory))

	XMOL = np.zeros((nframe, nmol))
	YMOL = np.zeros((nframe, nmol))
	ZMOL = np.zeros((nframe, nmol))
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

		if nsite > 1: XMOL[frame], YMOL[frame], ZMOL[frame] = molecules(xat, yat, zat, nmol, nsite, mol_M, mol_com)
		else: XMOL[frame], YMOL[frame], ZMOL[frame] = xat, yat, zat
			

	file_name_pos = file_name + '_{}'.format(nframe)

	print '\nSAVING OUTPUT MOLECULAR POSITION FILES\n'
	with file('{}/pos/{}_xmol.npy'.format(directory, file_name_pos), 'w') as outfile:
		np.save(outfile, XMOL)
	with file('{}/pos/{}_ymol.npy'.format(directory, file_name_pos), 'w') as outfile:
		np.save(outfile, YMOL)
	with file('{}/pos/{}_zmol.npy'.format(directory, file_name_pos), 'w') as outfile:
		np.save(outfile, ZMOL)
	with file('{}/pos/{}_com.npy'.format(directory, file_name_pos), 'w') as outfile:
		np.save(outfile, COM)


def read_mol_positions(directory, file_name, ntraj, nframe):

	file_name = '{}_{}'.format(file_name, ntraj)

	xmol = np.load('{}/pos/{}_xmol.npy'.format(directory, file_name), mmap_mode='r')[:nframe]
	ymol = np.load('{}/pos/{}_ymol.npy'.format(directory, file_name), mmap_mode='r')[:nframe]
	zmol = np.load('{}/pos/{}_zmol.npy'.format(directory, file_name), mmap_mode='r')[:nframe]

	return xmol, ymol, zmol

def read_com_positions(directory, file_name, ntraj, nframe):

	file_name = '{}_{}'.format(file_name, ntraj)

	COM = np.load('{}/pos/{}_com.npy'.format(directory, file_name), mmap_mode = 'r')[:nframe]

	return COM


def model_mdtraj():

	natom = int(traj.n_atoms)
	nmol = int(traj.n_residues)
	nsite = natom / nmol

	AT = [atom.name for atom in traj.topology.atoms][:nsite]
	M = [atom.mass for atom in traj.topology.atoms][:nsite] 
	sigma_m = float(raw_input("Enter molecular radius (Angstroms): "))

	return nsite, AT, M, sigma_m


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

	vector = np.array(vector)
	magnitude_2 = np.sum(vector.T**2, axis=axis)
	u_vector = np.sqrt(vector**2 / magnitude_2) * np.sign(vector)

	return u_vector


def normalise(A):

	max_A = np.max(A)
	min_A = np.min(A)

	return (np.array(A) - min_A) / (max_A - min_A)


def get_fourier(auv, nm):

        f = np.zeros(len(auv), dtype=complex)

        for u in xrange(-nm,nm+1):
                for v in xrange(-nm, nm+1):
                        index = (2 * nm + 1) * (u + nm) + (v + nm)

                        j1 = (2 * nm + 1) * (abs(u) + nm) + (abs(v) + nm)
                        j2 = (2 * nm + 1) * (-abs(u) + nm) + (abs(v) + nm)
                        j3 = (2 * nm + 1) * (abs(u) + nm) + (-abs(v) + nm)
                        j4 = (2 * nm + 1) * (-abs(u) + nm) + (-abs(v) + nm)

			if abs(u) + abs(v) == 0: f[index] = auv[j1]

                        elif v == 0: f[index] = (auv[j1] - np.sign(u) * 1j * auv[j2]) / 2.
                        elif u == 0: f[index] = (auv[j1] - np.sign(v) * 1j * auv[j3]) / 2.

                        elif u < 0 and v < 0: f[index] = (auv[j1] + 1j * (auv[j2] + auv[j3]) - auv[j4]) / 4.
                        elif u > 0 and v > 0: f[index] = (auv[j1] - 1j * (auv[j2] + auv[j3]) - auv[j4]) / 4.

                        elif u < 0: f[index] = (auv[j1] + 1j * (auv[j2] - auv[j3]) + auv[j4]) / 4.
                        elif v < 0: f[index] = (auv[j1] - 1j * (auv[j2] - auv[j3]) + auv[j4]) / 4.

        return f


def linear(x, m, c): return m*x + c


def gaussian(x, mean, std): return np.exp(-(x-mean)**2 / (2 * std**2)) / (SQRT2 * std * SQRTPI)


def laplace(x, mean, std): return np.exp(-abs(x-mean) / std ) / (2 * std)


def gaussian_smoothing(arrays, centres, deltas, DIM, nslice):

	cw_arrays = np.zeros((len(arrays), len(arrays[0])))

	stds = np.sqrt(np.array(deltas))
	lslice = DIM[2] / nslice
	Z1 = np.linspace(0, DIM[2], nslice)
	max_std = np.max(stds)
	length = int(max_std / lslice) * 12
	ZG = np.arange(-lslice*length/2, lslice*length/2 + lslice/2, lslice)
	P_arrays = [[gaussian(z, 0, STD) for z in ZG ] for STD in stds]

	#print ""
	#print max_std * 12, length, ZG[0], ZG[-1], lslice*length/2, ZG[1] - ZG[0], lslice

	for n1, z1 in enumerate(Z1):
		for n2, z2 in enumerate(ZG):
			sys.stdout.write("PERFORMING GAUSSIAN SMOOTHING {0:.1%} COMPLETE \r".format(float(n1 * nslice + n2) / nslice**2) )
			sys.stdout.flush()

			indexes = [int((z1 - z2 - z0) / DIM[2] * nslice) % nslice for z0 in centres]

			for i, array in enumerate(arrays):
				try: cw_arrays[i][n1] += array[indexes[i]] * P_arrays[i][n2] * lslice
				except IndexError: pass
	return cw_arrays


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


def save_npy(directory, file_name, array, frame=False, mode='w'):
	"""
	save_npy(directory, file_name, array, frame, mode='w')

	General purpose algorithm to save an array to a npy file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	array:  array_like (float);
		Data array to be saved
	frame:  int (optional)
		Trajectory frame to save
	mode:  str (optional)
		Option to append write 'w', or read-write 'rw'	
	"""

	outfile = np.memmap('{}/{}.npy'.format(directory, file_name), dtype=array.dtype, mode=mode, shape=array.shape)
	if not frame: outfile[:] = array[:]
	else: outfile[frame] = array


def load_npy(directory, file_name, frames=[]):
	"""
	load_npy(directory, file_name, array, frame='all')

	General purpose algorithm to load an array from a npy file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	frame:  int, list (optional)
		Trajectory frames to load

	Returns
	-------

	array:  array_like (float);
		Data array to be loaded
	"""

	if len(frame) == 0: array = np.load('{}_{}.npy'.format(directory, file_name), mmap_mode = 'r')
	else: array = np.load('{}_{}.npy'.format(directory, file_name), mmap_mode = 'r')[frames]

	return array
