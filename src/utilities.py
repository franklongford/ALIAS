"""
*************** UTILITIES MODULE *******************

Generic functions to be used across multiple programs
within ALIAS.

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 24/11/2016 by Frank Longford
"""

import numpy as np
import subprocess, os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import mdtraj as md


def read_atom_mol_dim(f):

	DIM = np.zeros(3)

	if os.path.isfile('{}.out'.format(f)) == True:

		IN = open('{}.out'.format(f))
		lines = IN.readlines()
		IN.close()

		for i in xrange(len(lines)):
			if lines[i].isspace() == False:
				temp_lines = lines[i].split()
				if temp_lines[0] == "NATOM":
					natom = int(temp_lines[2])

				if temp_lines[0] == "NHPARM":
					nmol = int(temp_lines[11])
					break

	else: print "{}.out FILE MISSING".format(f)

	if os.path.isfile('{}.rst'.format(f)) == True:

		IN = open('{}.rst'.format(f))
		lines = IN.readlines()
		IN.close()

		temp_lines = lines[-1].split()
		DIM[0] = float(temp_lines[0])
		DIM[1] = float(temp_lines[1])
		DIM[2] = float(temp_lines[2])					


		return natom, nmol, DIM

	else: print "{}.rst FILE MISSING".format(f)


def read_positions(fi, ncharge):
	"OPENS FILE AND RETURNS POSITIONS OF ATOMS AS X Y Z ARRAYS"
	FILE = open('{}'.format(fi), 'r')
	lines = FILE.readlines()
	FILE.close()

	l = len(lines)

	temp = lines[1].split()

	natom = int(temp[0])
	nmol = natom / ncharge
	ndof = natom * 3

	x = []
	y = []
	z = []

	nline = int(ndof/6) + 2

	"Loops through .rst file to copy atomic positions in rows of 6 positions long"
	for i in range(l)[2:nline]:

		temp_lines = lines[i].split()
	
		x.append(float(temp_lines[0]))
		x.append(float(temp_lines[3]))

		y.append(float(temp_lines[1]))
		y.append(float(temp_lines[4]))

		z.append(float(temp_lines[2]))
		z.append(float(temp_lines[5]))
	
	"Checks if there is a half row in the .rst file (3 positions long)"
	if np.mod(ndof,6) != 0:
		
		temp_lines = lines[nline].split()
	
		x.append(float(temp_lines[0]))
		y.append(float(temp_lines[1]))
		z.append(float(temp_lines[2]))


	return x, y, z

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


def get_param(model):

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
		LJ = [(0.07112, 2.782)]
	elif model.upper() == 'METHANOL':
		nsite = 6
		AT = ['H', 'C', 'H', 'H', 'O', 'H']
		Q = [6.77869560E-01, 2.12472018E+00, 6.77869560E-01, 6.77869560E-01, -1.18390283E+01, 7.68069945E+00]
		M = [1.00800000E+00, 1.20100000E+01, 1.00800000E+00, 1.00800000E+00, 1.60000000E+01, 1.00800000E+00]
		LJ = [(0.0150, 0.2104, 0.1094), (2.4535, 3.0665, 3.3997)]

	return nsite, AT, Q, M, LJ


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


def unit_vector(x, y, z):

	v = x**2 + y**2 + z**2
	c = 1./v

	ux = np.sqrt(x**2 * c) * np.sign(x)
	uy = np.sqrt(y**2 * c) * np.sign(y)
	uz = np.sqrt(z**2 * c) * np.sign(z)

	return ux, uy, uz


def local_frame_surface(dzx, dzy, z, zcom):
	
	if z < zcom: 
		t = unit_vector(1, 0, dzx)
		if np.any(np.isnan(t)) == True: print t
		n = unit_vector(0, 1, dzy)
		if np.any(np.isnan(n)) == True: print n
		d = unit_vector(-dzx, -dzy, 1)
		if np.any(np.isnan(d)) == True: print d
	else: 
		t = unit_vector(1, 0, -dzx)
		if np.any(np.isnan(t)) == True: print t
		n = unit_vector(0, 1, -dzy)
		if np.any(np.isnan(n)) == True: print n
		d = unit_vector(dzx, dzy, 1)
		if np.any(np.isnan(d)) == True: print d

	#d = np.cross(t, n)

	B = np.array([[t[0], n[0], d[0]], [t[1], n[1], d[1]], [t[2], n[2], d[2]]])
	
	return B


def read_energy_temp_tension(f):
	"Opens .out file and returns energy, temperature and surface tension of simulation"

	if os.path.isfile('{}.out'.format(f)) == True:

		FILE = open('{}.out'.format(f), 'r')
		lines = FILE.readlines()
		FILE.close()

		l = len(lines)
		ENERGY = []
		TENSION = []
		exit = 0
		j = 0
		n = 0

		average_line = l

		for n in xrange(l):
			if lines[n].isspace() == 0:
				temp_lines = lines[n].split()
				if temp_lines[0] == 'Etot' and n < average_line: ENERGY.append(float(temp_lines[2]))
				if temp_lines[0] == 'SURFTEN' and n < average_line: TENSION.append(float(temp_lines[2]))

				if temp_lines[0] == 'A' and temp_lines[1] == 'V' and len(temp_lines) > 1:
					average_line = n
					temp_lines = lines[n+3].split()
					temperature = float(temp_lines[8])

					temp_lines = lines[n+4].split()
					energy = float(temp_lines[2])

					temp_lines = lines[n+8].split()
					tension = float(temp_lines[2])

				if temp_lines[0] == 'R' and temp_lines[1] == 'M' and len(temp_lines) > 1:
					temp_lines = lines[n+3].split()
					t_err = float(temp_lines[8])

					temp_lines = lines[n+4].split()
					e_err = float(temp_lines[2])
			
					temp_lines = lines[n+8].split()
					rms = float(temp_lines[2])

					break

		try: return energy, e_err, temperature, t_err, tension, rms, ENERGY, TENSION
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


def ST_janecek(rhoz, rc, sigma, epsilon, dz):
	
	nslice = len(rhoz)
	zz_sum = 0.0
	xx_sum = 0.0
	gamma_sum = 0.0
	force = np.zeros(nslice)
	den_2 = np.zeros(nslice)
	total = np.zeros(nslice)

	for i in xrange(nslice):
		z = i * dz
		for j in xrange(nslice):
			ddz = z - (j * dz)
			xx = pi_xx(abs(ddz), rc, sigma) 
			zz = pi_zz(abs(ddz), rc, sigma) * 2
			gamma_sum += rhoz[i] * rhoz[j] * (zz - xx) * dz**2 * np.pi * epsilon

	return gamma_sum


def pi_xx(ddz, rc, sigma):
		
	if abs(ddz) <= rc: return ((6*rc**2-5*ddz**2)*(sigma/rc)**12 /5. - (3*rc**2 - 2*ddz**2) * (sigma/rc)**6 /2.)
	else: return  (ddz**2./5*(sigma/ddz)**12 - ddz**2/2.*(sigma/ddz)**6)


def pi_zz(ddz, rc, sigma):
		
	if abs(ddz) <= rc: return ddz**2 * ((sigma/rc)**12 - (sigma/rc)**6)
	else: return ddz**2 * ((sigma/ddz)**12 - (sigma/ddz)**6)


def E_janecek(rhoz, rc, sigma, epsilon, dz, A):
	
	nslices = len(rhoz)
	esum = 0.0
	
	for i in xrange(nslices):
		z = i * dz
		for j in xrange(nslices):
			ddz = z - (j * dz)
			esum += lambda_en(ddz, rc, sigma) * rhoz[j] * rhoz[i] * dz **2 * np.pi * epsilon * A / 2
		
	return esum

def lambda_en(ddz, rc, sigma):
		
	if abs(ddz) <= rc: return rc**2 * (2./5*(sigma/rc)**12 - (sigma/rc)**6)
	else: return ddz**2 * (2./5*(sigma/ddz)**12 - (sigma/ddz)**6)



def make_nc(directory, folder, model, size, suffix, N, del_rst):

	rst = md.formats.AmberRestartFile('{}/{}/{}_{}_{}0.rst'.format(directory, folder, model.lower(), size, suffix), mode='r')
	top = md.load_prmtop('{}/{}_{}.prmtop'.format(directory, model.lower(), size))
	main_traj = rst.read_as_traj(top)
	for j in xrange(1, N):
		rst = md.formats.AmberRestartFile('{}/{}/{}_{}_{}{}.rst'.format(directory, folder, model.lower(), size, suffix, j), mode='r')
		traj = rst.read_as_traj(top)
		main_traj = main_traj.join(traj)

	main_traj.save_netcdf('{}/{}/{}_{}_{}_{}.nc'.format(directory, folder, model.lower(), size, suffix, N))

	if del_rst == 'Y':
		for j in xrange(0, N):
			proc = subprocess.Popen('rm {}/{}/{}_{}_{}{}.rst'.format(directory, folder, model.lower(), size, suffix, j), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			out, err = proc.communicate()


def load_nc(directory, folder):

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

        f = np.zeros(len(auv))

        for u in xrange(-nm,nm+1):
                for v in xrange(-nm, nm+1):
                        index = (2 * nm + 1) * (u + nm) + (v + nm)

                        j1 = (2 * nm + 1) * (abs(u) + nm) + (abs(v) + nm)
                        j2 = (2 * nm + 1) * (-abs(u) + nm) + (abs(v) + nm)
                        j3 = (2 * nm + 1) * (abs(u) + nm) + (-abs(v) + nm)
                        j4 = (2 * nm + 1) * (-abs(u) + nm) + (-abs(v) + nm)

                        if u == 0: f[index] = (auv[j1] - np.sign(v) * 1j * auv[j3]) / 2.

                        elif v == 0: f[index] = (auv[j1] - np.sign(u) * 1j * auv[j2]) / 2.

                        elif u < 0 and v < 0: f[index] = (auv[j1] + 1j * (auv[j2] + auv[j3]) - auv[j4]) / 4.
                        elif u > 0 and v > 0: f[index] = (auv[j1] - 1j * (auv[j2] + auv[j3]) - auv[j4]) / 4.

                        elif u < 0: f[index] = (auv[j1] + 1j * (auv[j2] - auv[j3]) + auv[j4]) / 4.
                        elif v < 0: f[index] = (auv[j1] - 1j * (auv[j2] - auv[j3]) + auv[j4]) / 4.

        return f
	
