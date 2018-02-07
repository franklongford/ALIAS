"""
*************** UTILITIES MODULE *******************

Generic functions to be used across multiple programs
within ALIAS.

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford, Sam Munday

Last modified 29/11/2016 by Frank Longford
"""

import numpy as np
import subprocess, os, sys, pickle, tables

from scipy import stats
import scipy.constants as con

import mdtraj as md

import matplotlib.pyplot as plt

SQRT2 = np.sqrt(2.)
SQRTPI = np.sqrt(np.pi)


def make_checkfile(checkfile_name):

	checkfile = {}
	with file('{}.pkl'.format(checkfile_name), 'wb') as outfile:
		pickle.dump(checkfile, outfile, pickle.HIGHEST_PROTOCOL)

def read_checkfile(checkfile_name):

	with file('{}.pkl'.format(checkfile_name), 'rb') as infile:
        	return pickle.load(infile)

def update_checkfile(checkfile_name, symb, obj):

	with file('{}.pkl'.format(checkfile_name), 'rb') as infile:
        	checkfile = pickle.load(infile)
	checkfile[symb] = obj
	with file('{}.pkl'.format(checkfile_name), 'wb') as outfile:
        	pickle.dump(checkfile, outfile, pickle.HIGHEST_PROTOCOL)
	return checkfile


def get_sim_param(traj_dir, top_dir, traj_file, top_file):
	"""
	Returns selected parameters of input trajectory and topology file using mdtraj
	
	Keyword arguments:
	top_dir -- Directory of topology file
	traj_dir -- Directory of trajectory file
	traj_file -- Trajectory file name
	top_file -- Topology file name

	Output:
	traj -- mdtraj trajectory object
	MOL -- List of residue types in simulation cell
	nframe -- Number of frames sampled in traj_file
	dim -- Simulation cell dimensions (angstroms)
	"""

	traj = md.load('{}/{}'.format(traj_dir, traj_file), top='{}/{}'.format(top_dir, top_file))
	MOL = list(set([molecule.name for molecule in traj.topology.residues]))
	nframe = int(traj.n_frames)
	dim = np.array(traj.unitcell_lengths[0]) * 10

	return traj, MOL, nframe, dim


def molecules(xat, yat, zat, nmol, nsite, mol_M, mol_com):
	"RETURNS X Y Z ARRAYS OF MOLECULAR POSITIONS" 
	
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


def make_at_mol_com(traj, directory, file_name, natom, nmol, at_index, nframe, dim, nsite, M, mol, mol_com):
	

	print "\n-----------CREATING POSITIONAL FILES------------\n"

	if not os.path.exists("{}/pos".format(directory)): os.mkdir("{}/pos".format(directory))

	XAT = np.zeros((nframe, natom))
	YAT = np.zeros((nframe, natom))
	ZAT = np.zeros((nframe, natom))
	COM = np.zeros((nframe, 3))

	if nsite > 1:
		XMOL = np.zeros((nframe, nmol))
		YMOL = np.zeros((nframe, nmol))
		ZMOL = np.zeros((nframe, nmol))

	mol_M = np.array(M * nmol)
	mol_M /= mol_M.sum()

	XYZ = np.moveaxis(traj.xyz, 1, 2) * 10

	for frame in xrange(nframe):
		sys.stdout.write("PROCESSING {} out of {} IMAGES\r".format(frame, nframe) )
		sys.stdout.flush()

		COM[frame, :] = traj.xyz[frame].astype('float64').T.dot(mol_M) * 10

		XAT[frame] += XYZ[frame][0] 
		YAT[frame] += XYZ[frame][1] 
		ZAT[frame] += XYZ[frame][2]

		xat = XAT[frame][at_index]
		yat = YAT[frame][at_index]
		zat = ZAT[frame][at_index]

		if nsite > 1: XMOL[frame], YMOL[frame], ZMOL[frame] = molecules(xat, yat, zat, nmol, nsite, mol_M, mol_com)

	file_name_pos = file_name + '_{}'.format(nframe)

	print '\nSAVING OUTPUT POSITION FILES\n'
	with file('{}/pos/{}_xat.npy'.format(directory, file_name_pos), 'w') as outfile:
		np.save(outfile, XAT)
	with file('{}/pos/{}_yat.npy'.format(directory, file_name_pos), 'w') as outfile:
		np.save(outfile, YAT)
	with file('{}/pos/{}_zat.npy'.format(directory, file_name_pos), 'w') as outfile:
		np.save(outfile, ZAT)
	if nsite > 1:
		with file('{}/pos/{}_xmol.npy'.format(directory, file_name_pos), 'w') as outfile:
			np.save(outfile, XMOL)
		with file('{}/pos/{}_ymol.npy'.format(directory, file_name_pos), 'w') as outfile:
			np.save(outfile, YMOL)
		with file('{}/pos/{}_zmol.npy'.format(directory, file_name_pos), 'w') as outfile:
			np.save(outfile, ZMOL)
		with file('{}/pos/{}_com.npy'.format(directory, file_name_pos), 'w') as outfile:
			np.save(outfile, COM)


def read_atom_positions(directory, file_name, ntraj, nframe):

	file_name = '{}_{}'.format(file_name, ntraj)

	xat = np.load('{}/pos/{}_xat.npy'.format(directory, file_name), mmap_mode='r')[:nframe]
	yat = np.load('{}/pos/{}_yat.npy'.format(directory, file_name), mmap_mode='r')[:nframe]
	zat = np.load('{}/pos/{}_zat.npy'.format(directory, file_name), mmap_mode='r')[:nframe]

	return xat, yat, zat

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


def make_earray(file_name, arrays, atom, sizes):

	with tables.open_file(file_name, 'w') as outfile:
		for i, array in enumerate(arrays):
			outfile.create_earray(outfile.root, array, atom, sizes[i])


def radial_dist(root, directory, data_dir, traj_file, top_file, nsite, M, com, ow_pos):

	traj_file = raw_input("Enter cube trajectory file: ")
	file_end = max([0] + [pos for pos, char in enumerate(traj_file) if char == '/'])
	directory = traj_file[:file_end]
	traj_file = traj_file[file_end+1:]
	data_dir = directory + '/data'

	print directory, data_dir, traj_file

	traj = md.load('{}/{}'.format(directory, traj_file), top='{}/{}'.format(root, top_file))
	natom = int(traj.n_atoms)
	nmol = int(traj.n_residues)
	ntraj = int(traj.n_frames)
	DIM = np.array(traj.unitcell_lengths[0]) * 10

	lslice = 0.01
	max_r = np.min(DIM) / 2.
	nslice = int(max_r / lslice)

	new_XYZ = np.zeros((ntraj, natom, 3))

	XYZ = np.moveaxis(traj.xyz, 1, 2) * 10

	file_name = '{}_{}_{}'.format(top_file.split('.')[0], ntraj, com)
	if not os.path.exists('{}/pos/{}_zmol.npy'.format(data_dir, file_name)): make_at_mol_com(traj, traj_file, data_dir, '{}_{}_{}'.format(top_file.split('.')[0], ntraj, com), natom, nmol, ntraj, DIM, nsite, M, com)

	xmol, ymol, zmol = read_mol_positions(data_dir, top_file.split('.')[0], ntraj, ntraj, com)

	new_XYZ = np.stack((xmol, ymol, zmol), axis=2) / 10

	new_XYZ = np.pad(new_XYZ, ((0, 0), (0, natom-nmol), (0, 0)), 'constant', constant_values=0)

	traj.xyz = new_XYZ
	
	pairs = []
	for i in xrange(nmol): 
		for j in xrange(i): pairs.append([i, j])

	r, g_r = md.compute_rdf(traj[:10], pairs = pairs, bin_width = lslice/10, r_range = (0, max_r/10))
	plt.plot(r*10, g_r)

	mol_sigma = 2**(1./6) * g_r.argmax() * lslice

	print "r_max = {}    molecular sigma = {}".format(g_r.argmax()*lslice, mol_sigma)
	plt.show()


def model_mdtraj():

	natom = int(traj.n_atoms)
	nmol = int(traj.n_residues)
	nsite = natom / nmol

	AT = [atom.name for atom in traj.topology.atoms][:nsite]
	M = [atom.mass for atom in traj.topology.atoms][:nsite] 
	sigma_m = float(raw_input("Enter molecular radius (Angstroms): "))

	return nsite, AT, M, sigma_m


def centre_mass(xat, yat, zat, nsite, M):
	"Returns the coordinates of the centre of mass"

	nmol = len(xat) / nsite
	mol_M = np.array(M * nmol)

	xR = np.sum(xat * mol_M) / (nmol * np.sum(M))
	yR = np.sum(yat * mol_M) / (nmol * np.sum(M))
	zR = np.sum(zat * mol_M) / (nmol * np.sum(M))

	return xR, yR, zR


def bubblesort(alist, key):
	"Sorts arrays 'alist' and 'key' by order of elements of 'key'"
	for passnum in range(len(alist)-1,0,-1):
		for i in range(passnum):
			if key[i]>key[i+1]:
				temp = alist[i]
				alist[i] = alist[i+1]
				alist[i+1] = temp

				temp = key[i]
				key[i] = key[i+1]
				key[i+1] = temp


def normalise(A):
	max_A = np.max(A)
	min_A = np.min(A)
	return (np.array(A) - min_A) / (max_A - min_A)


def get_histograms(root, nimage, nslice, Vslice):
	proc = subprocess.Popen('ls {}/DATA/INTDEN/*DEN.txt'.format(root), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = proc.communicate()
	files = out.split()
	length = len(root) + 6
	size = 0
	
	for i in xrange(len(files)):
		temp = files[i]
		temp = temp[length:-1]
		temp = temp.split('_')
		if int(temp[3]) > size and int(temp[3]) < nimage: 
			size = int(temp[3])
			k = i
	
	if size == 0: return 0, np.zeros(nslice), np.zeros(nslice), np.zeros(nslice)

	with file(files[k], 'r') as infile:
		av_mass_den, av_atom_den, av_mol_den= np.loadtxt(infile)
	
	mass_count = av_mass_den * size * Vslice * con.N_A * (1E-8)**3
	atom_count = [int(av_atom_den[j] * size * Vslice) for j in xrange(nslice)]
	mol_count = [int(av_mol_den[j] * size * Vslice) for j in xrange(nslice)]
	
	return size, mass_count, atom_count, mol_count


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


def check_uv(u, v):

	if abs(u) + abs(v) == 0: return 1.
	elif u * v == 0: return 2.
	else: return 1.


def sum_auv_2(auv, nm, qm):

	if qm == 0:	
		j = (2 * nm + 1) * nm + nm 
		return auv[j]

	sum_2 = 0
	for u in xrange(-qm, qm+1):
                for v in xrange(-qm, qm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)

			if abs(u) + abs(v) == 0 : sum_2 += auv[j]
			else: sum_2 += 1/4. * check_uv(u, v) * auv[j]

	return sum_2

def H_var_est(auv, nm, qm, DIM):

	if qm == 0:	
		j = (2 * nm + 1) * nm + nm 
		return 0#auv[j]

	sum_2 = 0
	for u in xrange(-qm, qm+1):
                for v in xrange(-qm, qm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)
			sum_2 += 4 * np.pi**4 * check_uv(u, v) * auv[j] * (u**4/DIM[0]**4 + v**4/DIM[1]**4 + 2 * u**2 * v**2 / (DIM[0]**2 * DIM[1]**2))

	return sum_2


def load_pt(directory, ntraj):

	proc = subprocess.Popen('ls {}/*PT.npy'.format(directory), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = proc.communicate()
	pt_files = out.split()

	length = len(directory) + 1
	pt_st = []
	ntb = 0
	k = 0
	
	for i in xrange(len(pt_files)):
		temp = pt_files[i][length:-4].split('_')
		if int(temp[1]) == ntraj and int(temp[2]) >= ntb: 
			k = i
			ntb = int(temp[2])
			
	try: 
		if os.path.exists(pt_files[k]):
			with file(pt_files[k], 'r') as infile:
				pt_st = np.load(infile)

	except IndexError: pass

	return pt_st, ntb


def block_error(A, ntb, s_ntb=2):
	
	var2 = [np.var(a) for a in A]

	pt = [[] for a in A]

	for tb in range(s_ntb, ntb):
		stdev1 = [np.var(blocks) for blocks in blocksav(A, tb)]
		for i in range(len(A)): pt[i].append(stdev1[i] * tb / var2[i])

	return pt


def blocksav(A, tb):
	nb = len(A[0])/tb
	blocks = np.zeros((len(A), nb))
	for i in range(nb):
		for j in range(len(A)):
			blocks[j][i] += np.mean(A[j][i*tb: (i+1)*tb])
	return blocks


def get_corr_time(pt, ntb):

	x = 1. / np.arange(2, ntb)
	y = 1. / np.array(pt)

        m, intercept, _, _, _ = stats.linregress(x,y)

	return 1. / intercept

def autocorr(A):
	
	n = len(A)
	var = np.var(A)
	A -= np.mean(A)
	result = np.correlate(A, A, mode='full')[-n:]
	return result / (np.arange(n, 0, -1) * var)


def get_block_error_auv(auv2_1, auv2_2, directory, model, csize, ntraj, ntb, ow_ntb):

	if os.path.exists('{}/DATA/ACOEFF/{}_{}_{}_{}_PT.npy'.format(directory, model.lower(), csize, ntraj, ntb)) and not ow_ntb:
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PT.npy'.format(directory, model.lower(), csize, ntraj, ntb), 'r') as infile:
			pt_auv1, pt_auv2 = np.load(infile)
	else: 
		old_pt_auv1, old_ntb = load_pt('{}/DATA/ACOEFF'.format(directory), ntraj)
		if old_ntb == 0 or ow_ntb: 
			pt_auv1 = block_error(auv2_1, ntb)
			pt_auv2 = block_error(auv2_2, ntb)
		elif old_ntb > ntb:
			with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PT.npy'.format(directory, model.lower(), csize, ntraj, old_ntb), 'r') as infile:
                        	pt_auv1, pt_auv2 = np.load(infile)
			pt_auv1 = pt_auv1[:ntb]
			pt_auv2 = pt_auv2[:ntb]
		elif old_ntb < ntb:
			with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PT.npy'.format(directory, model.lower(), csize, ntraj, old_ntb), 'r') as infile:
                        	pt_auv1, pt_auv2 = np.load(infile)

			pt_auv1 = np.concatenate((pt_auv1, block_error(auv2_1, ntb, old_ntb)))
                	pt_auv2 = np.concatenate((pt_auv2, block_error(auv2_2, ntb, old_ntb)))	
		
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PT.npy'.format(directory, model.lower(), csize, ntraj, ntb), 'w') as outfile:
        		np.savetxt(outfile, (pt_auv1, pt_auv2))

        M = len(auv2_1)

	plt.figure(2)
	plt.plot(pt_auv1)
	plt.plot(pt_auv2)
	plt.show()

	corr_time_auv1 = get_corr_time(pt_auv1, ntb)
	corr_time_auv2 = get_corr_time(pt_auv2, ntb)

	print corr_time_auv1, np.sqrt(corr_time_auv1 / M), np.std(auv2_1)

        m_err_auv1 = (np.std(auv2_1) * np.sqrt(corr_time_auv1 / M))
	m_err_auv2 = (np.std(auv2_2) * np.sqrt(corr_time_auv2 / M))

        return m_err_auv1, m_err_auv2


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



