"""
*************** DENSITY ANALYSIS MODULE *******************

Atomic, molecular and mass density profile of simulation 
trajectories.

***************************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 24/11/2016 by Frank Longford
"""

import numpy as np
import scipy as sp
import time, sys, os

from scipy import constants as con
from scipy.optimize import curve_fit

import utilities as ut


def density_profile(traj, root, ntraj, nslice, natom, model, nsite, AT, M, folder, csize, suffix, DIM, ow_all):
	"Saves atom, mol and mass profiles as well as parameters for a tanh mol density function, fitted to ntraj number of trajectory snapshots" 

	if not os.path.exists("{}/DATA/DEN".format(root)): os.mkdir("{}/DATA/DEN".format(root))
	if model.upper() == 'METHANOL': com = 'COM'
	else: com = 0
	print ""
	print "CALCULATING DENSITY {}".format(root)

	xR = np.zeros(ntraj)
	yR = np.zeros(ntraj)
	zR = np.zeros(ntraj)

	dist = 0

	mass_count = np.zeros(nslice)
	atom_count = np.zeros(nslice)
	mol_count = np.zeros(nslice)
	H_count = np.zeros(nslice)

	av_mass_den = np.zeros(nslice)
	av_atom_den = np.zeros(nslice)
	av_mol_den = np.zeros(nslice)
	av_H_den = np.zeros(nslice)

	avpl = []
	avpv = []
	avden = []
	avz0 = []
	
	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice)

	dz = DIM[2] / nslice
	Vslice = DIM[0] * DIM[1] * dz
	Acm = 1E-8
	start_image = 0

	for image in xrange(ntraj):
		if os.path.exists('{}/DATA/DEN/{}_{}_{}_{}_COUNT.txt'.format(root, model.lower(), csize, nslice, image)) and ow_all.upper() != 'Y':
			start_image = image + 1
			with file('{}/DATA/DEN/{}_{}_{}_{}_COUNT.txt'.format(root, model.lower(), csize, nslice, image)) as infile:
				mass_count, atom_count, mol_count, H_count = np.loadtxt(infile)

			av_mass_den += mass_count / (ntraj * Vslice * con.N_A * Acm**3)
			av_atom_den += atom_count / (ntraj * Vslice)
			av_mol_den += mol_count / (ntraj * Vslice)
			av_H_den += H_count / (ntraj * Vslice)


	for image in xrange(start_image, ntraj):
		sys.stdout.write("PROCESSING {} out of {} IMAGES\r".format(image+1, ntraj) )
		sys.stdout.flush()
		mass_count = np.zeros(nslice)
		atom_count = np.zeros(nslice)
		mol_count = np.zeros(nslice)
		H_count = np.zeros(nslice)

		ZYX = np.rot90(traj.xyz[image])
		zat = ZYX[0] * 10
		yat = ZYX[1] * 10
		xat = ZYX[2] * 10

		xmol, ymol, zmol = ut.molecules(xat, yat, zat, nsite, M, com=com)
		xR[image], yR[image], zR[image] = ut.centre_mass(xat, yat, zat, nsite, M)
		
		for n in xrange(natom):
			z = (zat[n]-zR[image] + 0.5*DIM[2])
			index_at = int(z * nslice / DIM[2]) % nslice
			m = n % nsite
			mass_count[index_at] += M[m]
			atom_count[index_at] += 1 
			if m == 0:
				z = (zmol[n/nsite]-zR[image] + 0.5*DIM[2])
				index_mol = int(z * nslice / DIM[2]) % nslice
				mol_count[index_mol] += 1
				
			if AT[m]== 'H': H_count[index_at] += 1

		av_mass_den += mass_count / (ntraj * Vslice * con.N_A * Acm**3)
		av_atom_den += atom_count / (ntraj * Vslice)
		av_mol_den += mol_count / (ntraj * Vslice)
		av_H_den += H_count / (ntraj * Vslice)

		with file('{}/DATA/DEN/{}_{}_{}_{}_COUNT.txt'.format(root, model.lower(), csize, nslice, image), 'w') as outfile:
			np.savetxt(outfile, (mass_count, atom_count, mol_count, H_count), fmt='%-12.6f')		
	
	print "\n"

	param, _ = curve_fit(ut.den_func, Z1, av_mass_den, [1., 0., DIM[2]/2., DIM[2]/4., 2.])
	param = np.absolute(param)

	print av_mass_den
	print param
	
	print "WRITING TO FILE..."

	with file('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, ntraj), 'w') as outfile:
		np.savetxt(outfile, (av_mass_den, av_atom_den, av_mol_den, av_H_den), fmt='%-12.6f')

	with file('{}/DATA/DEN/{}_{}_{}_{}_PAR.txt'.format(root, model.lower(), csize, nslice, ntraj), 'w') as outfile:
		np.savetxt(outfile, param, fmt='%-12.6f')

	with file('{}/DATA/DEN/{}_{}_{}_COM.txt'.format(root, model.lower(), csize, ntraj), 'w') as outfile:
		np.savetxt(outfile, (xR, yR, zR), fmt='%-12.6f')

	print "{} {} {} COMPLETE\n".format(root, model.upper(), csize)


def main(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, TYPE, folder, nfolder, suffix, ntraj):

	for i in xrange(nfolder):
		directory = '{}/{}_{}'.format(root, TYPE.upper(), i)
		if not os.path.exists('{}/{}/{}_{}_{}_{}.nc'.format(directory, folder.upper(), model.lower(), csize, suffix, 800)): 
			ut.make_nc(directory, folder.upper(),  model.lower(), csize, suffix, ntraj, 'Y')
		traj = ut.load_nc(directory, folder.upper())							
		directory = '{}/{}'.format(directory, folder.upper())

		natom = traj.n_atoms
		nmol = traj.n_residues
		DIM = np.array(traj.unitcell_lengths[0]) * 10
		sigma = np.max(LJ[1])
		lslice = 0.05 * sigma
		nslice = int(DIM[2] / lslice)

		if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))

		if os.path.exists('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, ntraj)):
			print "FILE FOUND '{}/DATA/DEN/{}_{}_{}_{}_DEN.txt".format(directory, model.lower(), csize, nslice, ntraj)
			overwrite = raw_input("OVERWRITE? (Y/N): ")
			if overwrite.upper() == 'Y': 
				ow_count = raw_input("OVERWRITE COUNT? (Y/N): ")	
				density_profile(traj, directory, ntraj, nslice, natom, model, nsite, AT, M, folder, csize, suffix, DIM, ow_count)	
		else: density_profile(traj, directory, ntraj, nslice, natom, model, nsite, AT, M, folder, csize, suffix, DIM, 'N')

