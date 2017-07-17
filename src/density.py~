"""
*************** DENSITY ANALYSIS MODULE *******************

Atomic, molecular and mass density profile of simulation 
trajectories.

density_array[0] 		= mass count
density_array[1:n_atom_types]   = atom count
density_array[-1] 		= mol count

***************************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 24/04/2017 by Frank Longford
"""

import numpy as np
import scipy as sp
import time, sys, os

from scipy import constants as con
from scipy.optimize import curve_fit

import utilities as ut
import matplotlib.pyplot as plt


def density_profile(directory, model, csize, suffix, nimage, natom, nmol, nsite, AT, M, COM, DIM, nslice, ow_count):
	"Saves atom, mol and mass profiles as well as parameters for a tanh mol density function, fitted to ntraj number of trajectory snapshots" 

	print "\nCALCULATING DENSITY {}".format(directory)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	av_density_array = [np.zeros(nslice) for n in range(2 + n_atom_types)]

	avpl = []
	avpv = []
	avden = []
	avz0 = []
	
	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice)

	vslice = DIM[0] * DIM[1] * DIM[2] / nslice
	Acm = 1E-8
	start_image = 0

	for image in xrange(nimage):
		if os.path.exists('{}/DATA/DEN/{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), csize, nslice, image)) and not ow_count:
			try:
				with file('{}/DATA/DEN/{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), csize, nslice, image)) as infile:
					count_array = np.loadtxt(infile)
				for i in xrange(2 + n_atom_types): av_density_array[i] += count_array[i] / (nimage * vslice)
				start_image = image + 1
			except IndexError: pass

	for image in xrange(start_image, nimage):
		sys.stdout.write("PROCESSING {} out of {} IMAGES\r".format(image, nimage) )
		sys.stdout.flush()
	
		count_array = [np.zeros(nslice) for n in range(2 + n_atom_types)]

		xat, yat, zat = ut.read_atom_positions(directory, model, csize, image)
		xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, image)
		xR, yR, zR = COM[image]

		for n in xrange(natom):
			m = n % nsite
			at_type = AT[m]

			z = (zat[n] -zR + 0.5 * DIM[2])
			index_at = int(z * nslice / DIM[2]) % nslice

			count_array[0][index_at] += M[m]
			count_array[1 + atom_types.index(at_type)][index_at] += 1

			if m == 0:
				z = (zmol[n/nsite] - zR + 0.5 * DIM[2])
				index_mol = int(z * nslice / DIM[2]) % nslice
				count_array[-1][index_mol] += 1

		with file('{}/DATA/DEN/{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), csize, nslice, image), 'w') as outfile:
			np.savetxt(outfile, (count_array), fmt='%-12.6f')		
	
		for i in xrange(2 + n_atom_types): av_density_array[i] += count_array[i] / (nimage * vslice)

	print "\n"

	av_density_array[0] = av_density_array[0] / (con.N_A * Acm**3)

	param, _ = curve_fit(ut.den_func, Z1, av_density_array[0], [1., 0., DIM[2]/2., DIM[2]/4., 2.])
	param = np.absolute(param)

	print "WRITING TO FILE..."

	with file('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, nimage), 'w') as outfile:
		np.savetxt(outfile, (av_density_array), fmt='%-12.6f')

	with file('{}/DATA/DEN/{}_{}_{}_{}_PAR.txt'.format(directory, model.lower(), csize, nslice, nimage), 'w') as outfile:
		np.savetxt(outfile, param, fmt='%-12.6f')

	print "{} {} {} COMPLETE\n".format(directory, model.upper(), csize)


def radial_dist(traj, directory, model, nimage, max_r, lslice, nslice, natom, nmol, nsite, AT, M, csize, DIM, com, ow_count):


	print "CALCULATING RADIAL DISTRIBUTION {}".format(directory)

	dist = 0
	suffix = 'cube'

	periodic_images = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)
	V = DIM[0] * DIM[1] * DIM[2]

	atom_count = np.array([nmol] + [AT.count(atom) * nmol for atom in atom_types])
	atom_den = [at_c / V for at_c in atom_count]

	print atom_den, atom_count, atom_types

	av_density_array = [np.zeros(nslice) for n in range(1 + n_atom_types)]

	Acm = 1E-8
	start_image = 0

	for image in xrange(nimage):
		if os.path.exists('{}/DATA/DEN/{}_{}_{}_{}_RCOUNT.txt'.format(directory, model.lower(), csize, nslice, image)) and not ow_count:
			try:
				with file('{}/DATA/DEN/{}_{}_{}_{}_RCOUNT.txt'.format(directory, model.lower(), csize, nslice, image)) as infile:
					count_array = np.loadtxt(infile)
				for i in xrange(1 + n_atom_types): av_density_array[i] += count_array[i] / (atom_count[i] * float(nimage))
				start_image = image + 1
			except IndexError: pass

	for image in xrange(start_image, nimage):
		sys.stdout.write("PROCESSING {} out of {} IMAGES\r".format(image, nimage) )
		sys.stdout.flush()
	
		count_array = [np.zeros(nslice) for n in range(1 + n_atom_types)]

		ZYX = np.rot90(traj.xyz[image])
		zat = ZYX[0] * 10
		yat = ZYX[1] * 10
		xat = ZYX[2] * 10

		xmol, ymol, zmol = ut.molecules(xat, yat, zat, nsite, M, com=com)

		for n in xrange(natom):
			element1 = n % nsite
			molecule1 = int(n / nsite)
			at_type1 = AT[element1]
			for m in xrange(n):		
				element2 = m % nsite
				molecule2 = int(m / nsite)
				at_type2 = AT[element2]

				if molecule1 != molecule2 and at_type1 == at_type2:

					for im in periodic_images:
						rx = xat[n] - xat[m] + im[0] * DIM[0]
						ry = yat[n] - yat[m] + im[1] * DIM[1]
						rz = zat[n] - zat[m] + im[2] * DIM[2]

						r2 = rx**2 + ry**2 + rz**2

						if r2 < max_r**2:
							r = np.sqrt(r2)
							index_r = int(r * nslice / max_r) % nslice
							count_array[1 + atom_types.index(at_type1)][index_r] += 2

						if element1 + element2 == 0:	
							if com == '0': 
								if r2 < max_r**2: count_array[0][index_r] += 2
							else:
								rx = xmol[molecule1] - xmol[molecule2] + im[0] * DIM[0]
								ry = ymol[molecule1] - ymol[molecule2] + im[1] * DIM[1]
								rz = zmol[molecule1] - zmol[molecule2] + im[2] * DIM[2]

								r2 = rx**2 + ry**2 + rz**2

								if r2 < max_r**2:
									r = np.sqrt(r2)
									index_r = int(r * nslice / max_r) % nslice
									count_array[0][index_r] += 2

		with file('{}/DATA/DEN/{}_{}_{}_{}_RCOUNT.txt'.format(directory, model.lower(), csize, nslice, image), 'w') as outfile:
			np.savetxt(outfile, (count_array), fmt='%-12.6f')

		for i in xrange(1 + n_atom_types): av_density_array[i] += count_array[i] / (atom_count[i] * float(nimage))	
	
	print "\n"

	for j in xrange(nslice):
		r_ = (j + 0.5) * lslice
		vb = 4/3. * np.pi * ((r_ + lslice)**3 - r_**3) 
		for i in xrange(1 + n_atom_types): 
			av_density_array[i][j] = av_density_array[i][j] / (vb * atom_den[i]) 

	print "WRITING TO FILE..."

	with file('{}/DATA/DEN/{}_{}_{}_{}_RDEN.txt'.format(directory, model.lower(), csize, nslice, nimage), 'w') as outfile:
		np.savetxt(outfile, (av_density_array), fmt='%-12.6f')

	print "{} {} {} COMPLETE\n".format(directory, model.upper(), csize)



def density_thermo(traj, directory, model, csize, suffix, nimage, natom, nmol, nsite, AT, M, com, DIM, nslice, ow_count):
	"Saves atom, mol and mass profiles as well as parameters for a tanh mol density function, fitted to ntraj number of trajectory snapshots" 

	print "\nCALCULATING DENSITY {}".format(directory)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	av_density_array = [np.zeros(nslice) for n in range(2 + n_atom_types)]

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice)

	vslice = DIM[0] * DIM[1] * DIM[2] / nslice
	Acm = 1E-8
	start_image = 0

	for image in xrange(nimage):
		if os.path.exists('{}/DATA/DEN/{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), csize, nslice, image)) and not ow_count:
			try:
				with file('{}/DATA/DEN/{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), csize, nslice, image)) as infile:
					count_array = np.loadtxt(infile)
				for i in xrange(2 + n_atom_types): av_density_array[i] += count_array[i] / (nimage * vslice)
				start_image = image + 1
			except IndexError: pass

	for image in xrange(start_image, nimage):
		sys.stdout.write("PROCESSING {} out of {} IMAGES\r".format(image, nimage) )
		sys.stdout.flush()
	
		count_array = [np.zeros(nslice) for n in range(2 + n_atom_types)]

		ZYX = np.rot90(traj.xyz[image])
		zat = ZYX[0] * 10
		yat = ZYX[1] * 10
		xat = ZYX[2] * 10

		xmol, ymol, zmol = ut.molecules(xat, yat, zat, nsite, M, com=com)
		xR, yR, zR = ut.centre_mass(xat, yat, zat, nsite, M)

		for n in xrange(natom):
			m = n % nsite
			at_type = AT[m]

			z = (zat[n] -zR + 0.5 * DIM[2])
			index_at = int(z * nslice / DIM[2]) % nslice

			count_array[0][index_at] += M[m]
			count_array[1 + atom_types.index(at_type)][index_at] += 1

			if m == 0:
				z = (zmol[n/nsite] - zR + 0.5 * DIM[2])
				index_mol = int(z * nslice / DIM[2]) % nslice
				count_array[-1][index_mol] += 1

		with file('{}/DATA/DEN/{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), csize, nslice, image), 'w') as outfile:
			np.savetxt(outfile, (count_array), fmt='%-12.6f')		
	
		for i in xrange(2 + n_atom_types): av_density_array[i] += count_array[i] / (nimage * vslice)


	print "\n"

	av_density_array[0] = av_density_array[0] / (con.N_A * Acm**3)

	param, _ = curve_fit(ut.den_func, Z1, av_density_array[0], [1., 0., DIM[2]/2., DIM[2]/4., 2.])
	param = np.absolute(param)

	print "WRITING TO FILE..."

	with file('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, nimage), 'w') as outfile:
		np.savetxt(outfile, (av_density_array), fmt='%-12.6f')

	with file('{}/DATA/DEN/{}_{}_{}_{}_PAR.txt'.format(directory, model.lower(), csize, nslice, nimage), 'w') as outfile:
		np.savetxt(outfile, param, fmt='%-12.6f')

	print "{} {} {} COMPLETE\n".format(directory, model.upper(), csize)

