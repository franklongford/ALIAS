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
import subprocess, os, sys

from scipy import stats
import scipy.constants as con

import mdtraj as md

import matplotlib.pyplot as plt

SQRT2 = np.sqrt(2.)
SQRTPI = np.sqrt(np.pi)


def make_at_mol_com(traj, directory, model, csize, nsite, M, com):

	natom = int(traj.n_atoms)
	nmol = int(traj.n_residues)
	nframe = int(traj.n_frames)
	DIM = np.array(traj.unitcell_lengths[0]) * 10

	with file('{}/parameters.npy'.format(directory), 'w') as outfile:
		np.save(outfile, np.array([natom, nmol, nframe, DIM[0], DIM[1], DIM[2]]))

	if not os.path.exists("{}/POS".format(directory)): os.mkdir("{}/POS".format(directory))

	XAT = np.zeros((nframe, natom))
	YAT = np.zeros((nframe, natom))
	ZAT = np.zeros((nframe, natom))
	XMOL = np.zeros((nframe, nmol))
	YMOL = np.zeros((nframe, nmol))
	ZMOL = np.zeros((nframe, nmol))
	COM = np.zeros((nframe, 3))

	for frame in xrange(nframe):
		sys.stdout.write("PROCESSING {} out of {} IMAGES\r".format(frame, nframe) )
		sys.stdout.flush()

		XYZ = np.transpose(traj.xyz[frame])
		XAT[frame] += XYZ[0] * 10
		YAT[frame] += XYZ[1] * 10
		ZAT[frame] += XYZ[2] * 10

		XMOL[frame], YMOL[frame], ZMOL[frame] = molecules(XAT[frame], YAT[frame], ZAT[frame], nsite, M, com=com)
		COM[frame] = centre_mass(XAT[frame], YAT[frame], ZAT[frame], nsite, M)

		with file('{}/POS/{}_{}_{}_XAT.npy'.format(directory, model.lower(), csize, frame), 'w') as outfile:
			np.save(outfile, XAT[frame])
		with file('{}/POS/{}_{}_{}_YAT.npy'.format(directory, model.lower(), csize, frame), 'w') as outfile:
			np.save(outfile, YAT[frame])
		with file('{}/POS/{}_{}_{}_ZAT.npy'.format(directory, model.lower(), csize, frame), 'w') as outfile:
			np.save(outfile, ZAT[frame])
		with file('{}/POS/{}_{}_{}_XMOL.npy'.format(directory, model.lower(), csize, frame), 'w') as outfile:
			np.save(outfile, XMOL[frame])
		with file('{}/POS/{}_{}_{}_YMOL.npy'.format(directory, model.lower(), csize, frame), 'w') as outfile:
			np.save(outfile, YMOL[frame])
		with file('{}/POS/{}_{}_{}_ZMOL.npy'.format(directory, model.lower(), csize, frame), 'w') as outfile:
			np.save(outfile, ZMOL[frame])

	print 'SAVING OUTPUT FILES COM\n'
	with file('{}/POS/{}_{}_{}_COM.npy'.format(directory, model.lower(), csize, nframe), 'w') as outfile:
		np.save(outfile, COM)

	return natom, nmol, nframe, DIM, COM


def read_atom_positions(directory, model, csize, frame):

	file_name = '{}_{}_{}'.format(model.lower(), csize, frame)
	if os.path.exists('{}/POS/{}_XAT.txt'.format(directory, file_name)): convert_txt_npy('{}/POS/{}_XAT'.format(directory, file_name))
	if os.path.exists('{}/POS/{}_YAT.txt'.format(directory, file_name)): convert_txt_npy('{}/POS/{}_YAT'.format(directory, file_name))	
	if os.path.exists('{}/POS/{}_ZAT.txt'.format(directory, file_name)): convert_txt_npy('{}/POS/{}_ZAT'.format(directory, file_name))

	with file('{}/POS/{}_XAT.npy'.format(directory, file_name), 'r') as infile:
		xat = np.load(infile)
	with file('{}/POS/{}_YAT.npy'.format(directory, file_name), 'r') as infile:
		yat = np.load(infile)
	with file('{}/POS/{}_ZAT.npy'.format(directory, file_name), 'r') as infile:
		zat = np.load(infile)

	return xat, yat, zat

def read_mol_positions(directory, model, csize, frame):

	file_name = '{}_{}_{}'.format(model.lower(), csize, frame)
	if os.path.exists('{}/POS/{}_XMOL.txt'.format(directory, file_name)): convert_txt_npy('{}/POS/{}_XMOL'.format(directory, file_name))
	if os.path.exists('{}/POS/{}_YMOL.txt'.format(directory, file_name)): convert_txt_npy('{}/POS/{}_YMOL'.format(directory, file_name))	
	if os.path.exists('{}/POS/{}_ZMOL.txt'.format(directory, file_name)): convert_txt_npy('{}/POS/{}_ZMOL'.format(directory, file_name))

	with file('{}/POS/{}_XMOL.npy'.format(directory, file_name), 'r') as infile:
		xmol = np.load(infile)
	with file('{}/POS/{}_YMOL.npy'.format(directory, file_name), 'r') as infile:
		ymol = np.load(infile)
	with file('{}/POS/{}_ZMOL.npy'.format(directory, file_name), 'r') as infile:
		zmol = np.load(infile)

	return xmol, ymol, zmol

def convert_txt_npy(file_name):

	with file('{}.txt'.format(file_name), 'r') as infile: data = np.loadtxt(infile)
	with file('{}.npy'.format(file_name), 'w') as outfile: np.save(outfile, data)
	os.remove('{}.txt'.format(file_name))

def read_velocities(fi, nsite):
	"OPENS FILE AND RETURNS VELOCTIES OF ATOMS AS X Y Z ARRAYS"
	FILE = open('{}'.format(fi), 'r')
	lines = FILE.readlines()
	FILE.close()

	l = len(lines)

	temp = lines[1].split()

	natom = int(temp[0])
	nmol = natom / nsite
	ndof = natom * 3

	x = []
	y = []
	z = []

	start = int(ndof/6) + 2
	nline = 2 * int(ndof/6) + 2

	"Loops through .rst file to copy atomic velocities in rows of 6 velocities long"
	for i in range(l)[start:nline]:

		temp_lines = lines[i].split()
	
		x.append(float(temp_lines[0]))
		x.append(float(temp_lines[3]))

		y.append(float(temp_lines[1]))
		y.append(float(temp_lines[4]))

		z.append(float(temp_lines[2]))
		z.append(float(temp_lines[5]))
	
	"Checks if there is a half row in the .rst file (3 velocities long)"
	if np.mod(ndof,6) != 0:
		
		temp_lines = lines[nline].split()
	
		x.append(float(temp_lines[0]))
		y.append(float(temp_lines[1]))
		z.append(float(temp_lines[2]))

	return x, y, z


def get_model_param(model):

	if model.upper() == 'SPC':
		nsite = 3
		AT = ['O', 'H', 'H']
		Q = [-0.82, 0.41, 0.41]
		M = [1.60000000E+01, 1.00800000E+00, 1.00800000E+00]
		LJ = [0.1553, 3.166]
		
	elif model.upper() == 'SPCE':
		nsite = 3
		AT = ['O', 'H', 'H']
		Q = [-0.8476, 0.4238, 0.4238]
		M = [1.60000000E+01, 1.00800000E+00, 1.00800000E+00]
		LJ = [0.1553, 3.166]
	elif model.upper() == 'TIP3P':
		nsite = 3
		AT = ['O', 'H', 'H']
		Q = [-0.834, 0.417, 0.417]
		M = [1.60000000E+01, 1.00800000E+00, 1.00800000E+00]
		LJ = [0.1521, 3.15061]
	elif model.upper() == 'TIP4P':
		nsite = 4
		AT = ['O', 'H', 'H', 'lp']
		Q = [0.00000000E+00, 0.520, 0.52, -1.04]
		M = [1.60000000E+01, 1.00800000E+00, 1.00800000E+00, 0]
		LJ = [0.1550, 3.15365]
	elif model.upper() == 'TIP4P2005':
		nsite = 4
		AT = ['O', 'H', 'H', 'lp']
		Q = [0.00000000E+00, 0.5564, 0.5564, -2*0.5564]
		M = [1.60000000E+01, 1.00800000E+00, 1.00800000E+00, 0]
		LJ = [0.16275, 3.1589]
	elif model.upper() == 'TIP5P':
		nsite = 5
		AT = ['O', 'H', 'H', 'lp', 'lp']
		Q = [0.00000000E+00,  0.2410,   0.2410, - 0.2410, - 0.2410]
		M = [1.60000000E+01, 1.00800000E+00, 1.00800000E+00, 0, 0]
		LJ = [0.1600, 3.12000]
	elif model.upper() == 'ARGON':
		nsite = 1
		AT = ['Ar']
		Q = [0]
		M = [39.948]
		LJ = [0.2375, 3.40]
	elif model.upper() == 'NEON':
		nsite = 1
		AT = ['Ne']
		Q = [0]
		M = [20.180]
		LJ = [(0.07112), (2.782)]
	elif model.upper() == 'METHANOL':
		nsite = 6
		AT = ['H', 'C', 'H', 'H', 'O', 'H']
		Q = [0.0372, 0.1166, 0.0372, 0.0372, -0.6497, 0.4215]
		M = [1., 1.2E+01, 1., 1., 1.6E+01, 1.]
		LJ = [(0.0150, 0.2104, 0.1094), (2.4535, 3.0665, 3.3997)]
	elif model.upper() == 'ETHANOL':
		nsite = 9
		AT = ['H', 'C', 'H', 'H', 'C', 'H', 'H', 'O', 'H']
		Q = [0.0345, -0.099, 0.0345, 0.0345, -0.3118, -0.0294, -0.0294, -0.6418, 0.4143]
		M = [1.008, 1.2E+01, 1.008, 1.008, 1.2E+01, 1.008, 1.008, 1.6E+01, 1.008]
		LJ = [(0.0157, 0.2104, 0.1094, 0.0157), (2.47135, 3.06647, 3.39967, 2.64953)]
	elif model.upper() == 'DMSO':
		nsite = 10
		AT = ['H', 'C', 'H', 'H', 'S', 'O', 'C', 'H', 'H', 'H']
		Q =  [0.1423, -0.3244, 0.1423, 0.1423, 0.3155, -0.5205, -0.3244, 0.1423, 0.1423, 0.1423]
		M = [1.00800000E+00, 1.20100000E+01, 1.00800000E+00, 1.00800000E+00, 3.20600000E+01,  1.59900000E+01, 1.20100000E+01, 1.00800000E+00, 1.00800000E+00, 1.00800000E+00]
		LJ = [(0.2500, 0.2500, 0.2500, 0.2500), (3.56359, 3.39967, 2.9599, 2.47135)]
	elif model.upper() == 'AMOEBA':
		nsite = 3
		AT = ['O', 'H', 'H']
		Q = [False, False, False]
		M = [1.60000000E+01, 1.00800000E+00, 1.00800000E+00]
		LJ = [0.1553, 3.166]

	return nsite, AT, Q, M, LJ


def get_sim_param(root, directory, data_dir, model, nsite, suffix, csize, M, com, ow_pos, test):

	
	if os.path.exists('{}/parameters.nc'.format(data_dir)) and not ow_pos:	

		traj = md.load('{}/parameters.nc'.format(data_dir), top='{}/{}_{}.prmtop'.format(root, model.lower(), csize))
		natom = int(traj.n_atoms)
		nmol = int(traj.n_residues)
		if test: nframe = 6000
		else: nframe = 4000
		DIM = np.array(traj.unitcell_lengths[0]) * 10

		if os.path.exists('{}/POS/{}_{}_{}_COM.txt'.format(data_dir, model.lower(), csize, nframe)): convert_txt_npy('{}/POS/{}_{}_{}_COM'.format(data_dir, model.lower(), csize, nframe))
		print 'LOADING PARAMETER AND COM FILES'
		with file('{}/POS/{}_{}_{}_COM.npy'.format(data_dir, model.lower(), csize, nframe), 'r') as infile:
			COM = np.load(infile)
	else:
		traj = md.load('{}/{}_{}_{}.nc'.format(directory, model.lower(), csize, suffix), top='{}/{}_{}.prmtop'.format(root, model.lower(), csize))
		traj[0].save('{}/parameters.nc'.format(data_dir))
		natom, nmol, nframe, DIM, COM = make_at_mol_com(traj, data_dir, model, csize, nsite, M, com)

	return natom, nmol, nframe, DIM, COM


def get_thermo_constants(red_units, LJ):

	if red_units:
		e_constant = 1 / LJ[0]
		st_constant = ((LJ[1]*1E-10)**2) * con.N_A * 1E-6 / LJ[0]
		l_constant = 1 / LJ[1]
		p_constant = np.max(LJ[1])**3
		T_constant = con.k * con.N_A * 1E-3 / LJ[0] 
	else: 
		e_constant = 1.
		st_constant = 1.
		l_constant = 1E-10
		p_constant = 1.
		T_constant = 1.

	return e_constant, st_constant, l_constant, p_constant, T_constant


def get_polar_constants(model, a_type):

	water_au_A = 0.5291772083
	bohr_to_A = 0.529**3

	water_ame_a = [1.672, 1.225, 1.328]
	water_abi_a = [1.47, 1.38, 1.42]
	water_exp_a = [1.528, 1.415, 1.468]

	"Calculated polarisbilities taken from NIST Computational Chemistry Comparison and Benchmark DataBase (CCCBDB)"
	argon_exp_a = 1.642
	methanol_calc_a = [3.542, 3.0124, 3.073]  #B3LYP/aug-cc-pVQZ
	ethanol_calc_a = [5.648, 4.689, 5.027]  #B3LYP/Sadlej_pVTZ
	dmso_calc_a = [6.824, 8.393, 8.689]  #B3PW91/aug-cc-pVTZ

	if model.upper() == 'ARGON':	
		if a_type == 'exp': a = argon_exp_a
	elif model.upper() == 'METHANOL':
		if a_type == 'calc': a = methanol_calc_a
	elif model.upper() == 'ETHANOL':
		if a_type == 'calc': a = ethanol_calc_a
	elif model.upper() == 'DMSO':
		if a_type == 'calc': a = dmso_calc_a
	else:
		if a_type == 'exp': a = water_exp_a
		elif a_type == 'ame': a = water_ame_a
		elif a_type == 'abi': a = water_abi_a

	return a


def get_ism_constants(model, sigma):

	if model.upper() == 'ARGON':
		mol_sigma = sigma
		ns = 0.8
	elif model.upper() == 'SPCE':
		mol_sigma = sigma
		ns = 1.20
	if model.upper() == 'TIP4P2005':
		mol_sigma = sigma
		ns = 1.20
	elif model.upper() == 'METHANOL': 
		mol_sigma = 4.30
		ns = 1.20
	elif model.upper() == 'ETHANOL': 
		mol_sigma = 4.80
		ns = 1.20
	elif model.upper() == 'DMSO': 
		mol_sigma = 6.00
		ns = 1.20

	return mol_sigma, ns


def den_profile(zat, DIM, nslice):
	"RETURNS ATOMIC/MOLECULAR DENSITY PROFILE OF INPUT ARRAY"
	
	dz = DIM[2] / nslice
	Vslice = DIM[0] * DIM[1] * dz
	
	den = np.zeros(nslice)
	
	for i in xrange(len(zat)):
		index = int(zat[i] * nslice / DIM[2])
		den[index] += 1

	den = den / Vslice			
	
	return den


def molecules(xat, yat, zat, nsite, M, com='0'):
	"RETURNS X Y Z ARRAYS OF MOLECULAR POSITIONS" 
	
	nmol = len(xat) / nsite
	xmol = []
	ymol = []
	zmol = []	
	
	if com == 'COM':
		"USE CENTRE OF MASS AS MOLECULAR POSITION"
		M_sum = np.sum(M)
                for i in xrange(nmol):
                        index = i * nsite
                        xcom = 0.
                        ycom = 0.
                        zcom = 0.
                        for j in xrange(nsite):
                                xcom += M[j] * xat[index+j]
                                ycom += M[j] * yat[index+j]
                                zcom += M[j] * zat[index+j]
                        xmol.append(xcom/(M_sum))
                        ymol.append(ycom/(M_sum))
                        zmol.append(zcom/(M_sum))	
	
	else:
		"USE SINGLE ATOM AS MOLECULAR POSITION"
		com = int(com)
		for i in range(nmol):
			index = i * nsite
			xmol.append(xat[index])
			ymol.append(yat[index])
			zmol.append(zat[index])

	return xmol, ymol, zmol


def read_densities(root):
	import subprocess
	proc = subprocess.Popen('ls {}/DATA/*MD.txt'.format(root), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = proc.communicate()
	files = out.split()
	#print files
	length = len(root) + 6
	size = 0
	nimage = 0
	
	for i in xrange(len(files)):
		temp = files[i]
		temp = temp[length:-1]
		temp = temp.split('_')
		if int(temp[2]) > size and int(temp[3]) >= nimage: 
			size = int(temp[2])
			k = i
			nimage = int(temp[3])

	FILE = open('{}'.format(files[k]), 'r')
	lines = FILE.readlines()
	FILE.close()

	den = lines[-1].split()
	
	return float(den[0]), float(den[1]), float(den[2]), float(den[3]), float(den[4])


def read_profile(root, TYPE, nimage):
	import subprocess
	proc = subprocess.Popen('ls {}/DATA/*{}.txt'.format(root, TYPE), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = proc.communicate()
	files = out.split()

	length = len(root) + 6
	nslice = 0
	
	for i in xrange(len(files)):
		temp = files[i]
		temp = temp[length:-1]
		temp = temp.split('_')
		if int(temp[2]) > nslice and int(temp[3]) == nimage: 
			nslice = int(temp[2])
			k = i
	
	FILE = open('{}'.format(files[k]), 'r')
	lines = FILE.readlines()
	FILE.close()	

	DEN = np.zeros(nslice)

	for i in xrange(nslice):
		temp_lines = lines[i+2].split()
		DEN[i] = float(temp_lines[0])
		
	temp_lines = lines[1].split()
	DIM = [float(temp_lines[0]), float(temp_lines[1]), float(temp_lines[2])]
	den = lines[-1].split()
	PARAM = [float(den[0]), float(den[1]), float(den[2]), float(den[3]), float(den[4])]

	return nslice, DIM, DEN, PARAM


def den_func(z, pl, pv, z0, z1, d):
	return 0.5 * (pl + pv) - 0.5 * (pl - pv) * (np.tanh((z - (z0-z1))/ (2*d)) * np.tanh((z - (z0+z1))/ (2*d)))


def unit_vector(vector):

        x, y, z = vector

        v = x**2 + y**2 + z**2
        c = 1./v

        ux = np.sqrt(x**2 * c) * np.sign(x)
        uy = np.sqrt(y**2 * c) * np.sign(y)
        uz = np.sqrt(z**2 * c) * np.sign(z)

        return (ux, uy, uz)


def local_frame_molecule(molecule, model):

	if model.upper() == 'METHANOL':
		axx = np.subtract(molecule[1], molecule[4])
		ayy = np.cross(axx, np.subtract(molecule[5], molecule[1]))
		rot = np.array([unit_vector([ 3.4762,  0.0000,  -.6773]),
				unit_vector([ 0.0000,  3.0124,  0.0000]),
				unit_vector([ 0.5828,  0.0000,  3.0167])])
		rot = np.transpose(rot)
		azz = np.cross(ayy, axx)
		
	elif model.upper() == 'ETHANOL':
		CC = np.subtract(molecule[1], molecule[4])
		axx = np.subtract(molecule[7], molecule[1])
		ayy = np.cross(CC, axx)
		rot = np.array([unit_vector([ 5.4848,  0.0000, -1.3478]),
				unit_vector([ 0.0000,  4.6893, 0.0000]),
				unit_vector([ 1.1996,  0.0000, 4.8819])])
		azz = np.cross(ayy, axx)
	elif model.upper() == 'DMSO':
		c = np.subtract(molecule[1], molecule[6])
		b = np.add(np.subtract(molecule[4], molecule[1]), np.subtract(molecule[4],molecule[6]))
		rot = np.array([unit_vector([ 5.8192,  3.5643,  0.0000]),
				unit_vector([-4.3840,  7.1575,  0.0000]),
				unit_vector([ 0.0000,  0.0000,  8.6887])])
	else:
		azz = np.add(np.subtract(molecule[0], molecule[1]), np.subtract(molecule[0],molecule[2]))
		axx = np.subtract(molecule[1], molecule[2])
		rot = np.identity(3)
		ayy = np.cross(axx, azz)

	#if model.upper() != 'ETHANOL': a = np.cross(ayy, azz)

	axx = unit_vector(axx)
	ayy = unit_vector(ayy)
	azz = unit_vector(azz)

	O = np.array([axx, ayy, azz])
	O = np.transpose(O)
	O = np.dot(O, rot)

	return O


def local_frame_surface(dzx, dzy, ref):

        a = unit_vector([1, 0, dzx])
        if np.any(np.isnan(a)) == True: print a
        b = unit_vector([0, 1, dzy])
        if np.any(np.isnan(b)) == True: print b
        c = unit_vector([-ref * dzx, - ref * dzy, ref])
        if np.any(np.isnan(c)) == True: print c

	#a = unit_vector([1, 0, dzx])
	#c = unit_vector([-dzx, -dzy, 1])
	a = np.cross(b, c)

        B = np.array([a, b, c])

        return np.transpose(B)


def read_energy_temp_tension(f):
	"Opens .out file and returns energy, temperature and surface tension of simulation"

	if os.path.isfile('{}.out'.format(f)) == True:

		FILE = open('{}.out'.format(f), 'r')
		lines = FILE.readlines()
		FILE.close()

		l = len(lines)
		ENERGY = []
		POTENTIAL = []
		KINETIC = []
		TENSION = []
		TEMP = []
		exit = 0
		j = 0
		n = 0

		average_line = l

		for n in xrange(l):
			if lines[n].isspace() == 0:
				temp_lines = lines[n].split()
				if temp_lines[0] == 'NSTEP' and n < average_line: TEMP.append(float(temp_lines[8]))
				if temp_lines[0] == 'Etot' and n < average_line: 
					ENERGY.append(float(temp_lines[2]))
					KINETIC.append(float(temp_lines[5]))
					POTENTIAL.append(float(temp_lines[8]))
				if temp_lines[0] == 'SURFTEN' and n < average_line: TENSION.append(float(temp_lines[2]))

				if temp_lines[0] == 'A' and temp_lines[1] == 'V' and len(temp_lines) > 1:
					average_line = n
					temp_lines = lines[n+3].split()
					temperature = float(temp_lines[8])

					temp_lines = lines[n+4].split()
					energy = float(temp_lines[2])
					kinetic = float(temp_lines[5])
					potential = float(temp_lines[8])

					temp_lines = lines[n+8].split()
					tension = float(temp_lines[2])

				if temp_lines[0] == 'R' and temp_lines[1] == 'M' and len(temp_lines) > 1:
					temp_lines = lines[n+3].split()
					t_err = float(temp_lines[8])

					temp_lines = lines[n+4].split()
					e_rms = float(temp_lines[2])
					kin_rms = float(temp_lines[5])
					pot_rms = float(temp_lines[8])
			
					temp_lines = lines[n+8].split()
					temp_rms = float(temp_lines[2])

					break

		try: return energy, potential, kinetic, temperature, t_err, tension, ENERGY, POTENTIAL, KINETIC, TENSION, TEMP
		except UnboundLocalError: print "{}.out FILE INCOMPLETE".format(f)

	else: print "{}.out FILE MISSING".format(f)


def centre_mass(xat, yat, zat, nsite, M):
	"Returns the coordinates of the centre of mass"

	xR = 0
	yR = 0
	zR = 0
	
	m = 0

	for i in xrange(len(xat)):
		xR += M[i % nsite] * xat[i]
		yR += M[i % nsite] * yat[i]
		zR += M[i % nsite] * zat[i]
		m += M[i % nsite]
	
	xR = xR / m
	yR = yR / m
	zR = zR / m

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


def make_nc(directory, folder, model, size, suffix, N, del_rst):

	rst = md.formats.AmberRestartFile('{}/{}/{}_{}_{}0.rst'.format(directory, folder, model.lower(), size, suffix), mode='r')
	top = md.load_prmtop('{}/{}_{}.prmtop'.format(directory, model.lower(), size))
	main_traj = rst.read_as_traj(top)
	for j in xrange(1, N):
		sys.stdout.write("PROCESSING AMBER RESTART FILES  {} out of {} \r".format(j+1, N) )
		sys.stdout.flush()

		rst = md.formats.AmberRestartFile('{}/{}/{}_{}_{}{}.rst'.format(directory, folder, model.lower(), size, suffix, j), mode='r')
		traj = rst.read_as_traj(top)
		main_traj = main_traj.join(traj)

	main_traj.save_netcdf('{}/{}/{}_{}_{}.nc'.format(directory, folder, model.lower(), size, suffix))

	if del_rst:
		for j in xrange(0, N):
			proc = subprocess.Popen('rm {}/{}/{}_{}_{}{}.rst'.format(directory, folder, model.lower(), size, suffix, j), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			out, err = proc.communicate()


def load_nc(directory, folder, model, csize, suffix):

	traj = md.load('{}/{}/{}_{}_{}.nc'.format(directory, folder.upper(), model.lower(), csize, suffix), top='{}/{}_{}.prmtop'.format(directory, model.lower(), csize))
	"""
	proc = subprocess.Popen('ls {}/*.prmtop'.format(directory), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = proc.communicate()
	tops = out.split()

	proc = subprocess.Popen('ls {}/{}/*.nc'.format(directory, folder), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = proc.communicate()
	nc_files = out.split()

	length = len(directory) + len(folder) + 2
	nimage = 0
	
	for i in xrange(len(nc_files)):
		temp = nc_files[i][length:-3].split('_')
		if int(temp[3]) >= nimage: 
			k = i
			nimage = int(temp[3])
	
	traj = md.load(nc_files[k], top=tops[0])
	"""
	return traj


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

	proc = subprocess.Popen('ls {}/*PT.txt'.format(directory), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
				pt_st = np.loadtxt(infile)

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

	if os.path.exists('{}/DATA/ACOEFF/{}_{}_{}_{}_PT.txt'.format(directory, model.lower(), csize, ntraj, ntb)) and not ow_ntb:
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PT.txt'.format(directory, model.lower(), csize, ntraj, ntb), 'r') as infile:
			pt_auv1, pt_auv2 = np.loadtxt(infile)
	else: 
		old_pt_auv1, old_ntb = load_pt('{}/DATA/ACOEFF'.format(directory), ntraj)
		if old_ntb == 0 or ow_ntb: 
			pt_auv1 = block_error(auv2_1, ntb)
			pt_auv2 = block_error(auv2_2, ntb)
		elif old_ntb > ntb:
			with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PT.txt'.format(directory, model.lower(), csize, ntraj, old_ntb), 'r') as infile:
                        	pt_auv1, pt_auv2 = np.loadtxt(infile)
			pt_auv1 = pt_auv1[:ntb]
			pt_auv2 = pt_auv2[:ntb]
		elif old_ntb < ntb:
			with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PT.txt'.format(directory, model.lower(), csize, ntraj, old_ntb), 'r') as infile:
                        	pt_auv1, pt_auv2 = np.loadtxt(infile)

			pt_auv1 = np.concatenate((pt_auv1, block_error(auv2_1, ntb, old_ntb)))
                	pt_auv2 = np.concatenate((pt_auv2, block_error(auv2_2, ntb, old_ntb)))	
		
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PT.txt'.format(directory, model.lower(), csize, ntraj, ntb), 'w') as outfile:
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

