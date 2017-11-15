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

def count(directory, zat, zmol, model, nslice, nsite, AT, M, DIM, ow_count):
	
	natom = len(zat)
	nmol = len(zmol)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	count_array = np.zeros((2 + n_atom_types, nslice))
	at_index = np.array([atom_types.index(at) for at in AT] * nmol)
	
	z = zat + DIM[2]/2.
	index_z = np.array(z * nslice / DIM[2], dtype=int) % nslice

	for i in xrange(n_atom_types):
		count_array[i+1] += np.histogram(index_z[at_index == i], bins=nslice, range=(0, nslice))[0]
		count_array[0] += count_array[i+1] * M[AT.index(atom_types[i])]

	z = zmol + DIM[2]/2.
	index_z = np.array(z * nslice / DIM[2], dtype=int) % nslice
	count_array[-1] += np.histogram(index_z, bins=nslice, range=(0, nslice))[0]

	assert np.sum(count_array[-1]) == nmol

	return count_array


def density_profile(directory, model, csize, suffix, nframe, ntraj, natom, nmol, nsite, AT, M, com, DIM, nslice, ow_count):
	"Saves atom, mol and mass profiles as well as parameters for a tanh mol density function, fitted to ntraj number of trajectory snapshots" 

	print "\nCALCULATING DENSITY {}".format(directory)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	#print AT, M, atom_types
	
	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice)

	vslice = DIM[0] * DIM[1] * DIM[2] / nslice
	Acm = 1E-8

	file_name_den = '{}_{}_{}'.format(model.lower(), nslice, nframe)

	if not ow_count:
		try:
			with file('{}/DEN/{}_COUNT.npy'.format(directory, file_name_den)) as infile:
				tot_count_array = np.load(infile)
		except: ow_count = True

	if ow_count: 

		xat, yat, zat = ut.read_atom_positions(directory, model, csize, ntraj, nframe, com)
		xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, ntraj, nframe, com)
		COM = ut.read_com_positions(directory, model, csize, ntraj, nframe, com)
		tot_count_array = np.zeros((nframe, 2 + n_atom_types, nslice))  

		for frame in xrange(nframe):
			sys.stdout.write("PROCESSING {} out of {} FRAMES\r".format(frame, nframe) )
			sys.stdout.flush()

			file_name_count = '{}_{}_{}'.format(model.lower(), nslice, frame)

			if os.path.exists('{}/DEN/{}_COUNT.txt'.format(directory, file_name_count)): ut.convert_txt_npy('{}/DEN/{}_COUNT'.format(directory, file_name_count))
			if os.path.exists('{}/DEN/{}_COUNT.npy'.format(directory, file_name_count)):
				try:
					with file('{}/DEN/{}_COUNT.npy'.format(directory, file_name_count)) as infile:
						count_array = np.load(infile)
					os.remove('{}/DEN/{}_COUNT.npy'.format(directory, file_name_count))
				except IndexError: ow_count = True

			if ow_count: count_array = count(directory, zat[frame]-COM[frame][2], zmol[frame]-COM[frame][2], model, nslice, nsite, AT, M, DIM, ow_count)
		
			tot_count_array[frame] += count_array


		with file('{}/DEN/{}_COUNT.npy'.format(directory, file_name_den), 'w') as outfile:
			np.save(outfile, (tot_count_array))

	av_density_array = np.sum(tot_count_array, axis=0) / (nframe * vslice)
	av_density_array[0] = av_density_array[0] / (con.N_A * Acm**3)

	popt, perr = curve_fit(ut.den_func, Z1, av_density_array[0], [1., 0., DIM[2]/2., DIM[2]/4., 2.])
	popt = np.absolute(popt)

	"""
	plt.plot(Z1, av_density_array[-1])
	#plt.plot(Z1, [ut.den_func(z, popt[0], popt[1], popt[2], popt[3], popt[4]) for z in Z1])
	plt.show()
	plt.close('all')
	#"""

	print "\nWRITING TO FILE..."

	if os.path.exists('{}/DEN/{}_DEN.txt'.format(directory, file_name_den)):
		os.remove('{}/DEN/{}_DEN.txt'.format(directory, file_name_den))
		os.remove('{}/DEN/{}_PAR.txt'.format(directory, file_name_den))

	with file('{}/DEN/{}_DEN.npy'.format(directory, file_name_den), 'w') as outfile:
		np.save(outfile, (av_density_array))
	with file('{}/DEN/{}_PAR.npy'.format(directory, file_name_den), 'w') as outfile:
		np.save(outfile, popt)

	print "{} {} {} COMPLETE\n".format(directory, model.upper(), csize)


