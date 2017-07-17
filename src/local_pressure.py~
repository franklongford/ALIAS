"""
*************** LOCAL PRESSURE MODULE *******************



***************************************************************
Created 31/01/2017 by Frank Longford

Contributors: Frank Longford

Last modified 31/01/2017 by Frank Longford
"""

import numpy as np
import scipy as sp
import subprocess, time, sys, os, math, copy, gc
import matplotlib.pyplot as plt

from scipy import stats, interpolate, integrate
from scipy import constants as con
from scipy.optimize import curve_fit, leastsq
import scipy.integrate as spin
from scipy.interpolate import bisplrep, bisplev, splprep, splev
import matplotlib.pyplot as plt

import utilities as ut
import density as den
import mdtraj as md


def FORCE_VDW2(r2, sig, ep): return 24 * ep * (2 * (sig**12/r2**6) - (sig**6/r2**3))


def theta_step(x):
	if x < 0: return 1
	else: return 0


def gamma_calc(Z, rx, ry, rz, sig, ep, rc, DIM):

	N = len(rx)

	PT_sum = 0
	PN_sum = 0

	for i in xrange(N):
		for j in xrange(i):
			dx = rx[i] - rx[j]
			dx += DIM[0] * int(2 * dx / DIM[0])
			dy = ry[i] - ry[j]
			dy += DIM[1] * int(2 * dy / DIM[1])
			dz = rz[i] - rz[j]			
			r2 = dx**2 + dy**2 + dz**2
			if r2 <= rc**2:
				frc = FORCE_VDW2(r2, sig, ep) / r2
				#if frc / abs(dz) > 1000: print abs(dz), dz**2 * frc / abs(dz)
				PT_sum += (dx**2 + dy**2) / 2. * frc
				PN_sum += dz**2 * frc

	return (PN_sum - PT_sum) / (DIM[0]*DIM[1] * 2.) 
		

def PT_PN(Z, nslice, rx, ry, rz, sig, ep, rc, DIM):

	N = len(rx)

	PT_sum = np.zeros(nslice)
	PN_sum = np.zeros(nslice)

	for i in xrange(N):
		for j in xrange(i):
			dx = rx[i] - rx[j]
			dx += DIM[0] * int(2 * dx / DIM[0])
			dy = ry[i] - ry[j]
			dy += DIM[1] * int(2 * dy / DIM[1])
			dz = rz[i] - rz[j]			
			r2 = dx**2 + dy**2 + dz**2
			if r2 <= rc**2:
				for k, z in enumerate(Z):
					step = theta_step((z - rz[i]) / dz) * theta_step((rz[j] - z) / dz)
					if step != 0:
						frc = FORCE_VDW2(r2, sig, ep) / (abs(dz) * r2)
						if abs(frc) < 1000:
							PT_sum[k] += (dx**2 + dy**2) / 2 * frc 
							PN_sum[k] += dz**2 * frc
	
	return PT_sum, PN_sum			


def main(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, TYPE, folder, nfolder, suffix):

	rc = float(cutoff)

	"Conversion of length and surface tension units"
	if model.upper() == 'ARGON':
		#LJ[0] = LJ[0] * 4.184
		e_constant = 1 / (LJ[0] * 4.184)
                st_constant = ((LJ[1]*1E-10)**2) * con.N_A * 1E-6 / (LJ[0] * 4.184)
                l_constant = 1 / LJ[1]
		com = 0
	else: 
		#LJ[0] = LJ[0] * 4.184
		e_constant = 1.
                st_constant = 1.
                l_constant = 1E-10
		T = 298
		if model.upper() == 'METHANOL': com = 'COM'
		else: com = 0
	
	if suffix == 'surface': 
		sep = 100
		offset = 1
	elif suffix == 'velocity':
		sep = 1
		offset = 0

	if TYPE == 'C': csize = 30
	else: csize = 50

	ZA_RANGE = np.zeros(nfolder)
	surften_err = np.zeros(nfolder)
	MARKER = ['o', 'v', 'x', '^', 's', 'p', '+', '*']
	COLOUR = ['b', 'g', 'r', 'c', 'm', 'saddlebrown', 'navy']

	nfolder = 5

	for n in xrange(nfolder):
		if model.upper() == 'ARGON': directory = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/{}_TEST/{}_{}'.format(model.upper(), T, cutoff, TYPE.upper(), TYPE.upper(), n)
		else: directory = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A/{}_TEST/{}_{}'.format(model.upper(), T, cutoff, TYPE.upper(), TYPE.upper(), n)

		traj = ut.load_nc(directory, folder, model, csize, suffix)							
		directory = '{}/{}'.format(directory, folder.upper())
		print directory
		if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))
		if not os.path.exists("{}/DATA/PRESSURE".format(directory)): os.mkdir("{}/DATA/PRESSURE".format(directory))

		natom = traj.n_atoms
		nmol = traj.n_residues
		ntraj = traj.n_frames
		DIM = np.array(traj.unitcell_lengths[0]) * 10
		sigma = np.max(LJ[1])
		lslice = 0.05 * sigma
		nslice = int(DIM[2] / lslice)
		vlim = 3
		ncube = 3
		nm = int(DIM[0] / sigma)
		nxy = int((DIM[0] + DIM[1]) / sigma)

		TOT_PT = np.zeros(nslice)
		TOT_PN = np.zeros(nslice)
		Z = np.linspace(0, DIM[2], nslice)

		with file('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, ntraj), 'r') as infile:
			_, _, av_mol_den, _ = np.loadtxt(infile)

		ST = []
		ST1 = []

		conv = 4.184 * 1E26 / (con.N_A * 2)

		FILE = '{}/{}_{}_{}'.format(directory, model.lower(), csize, suffix)
		_, _, _, _ , _, _, TOTAL_ENERGY, TOTAL_TENSION, _ = ut.read_energy_temp_tension(FILE)	

		ow = 0

		for image in xrange(ntraj):
			print "Calculating Pressure Tensor, Image {}".format(image)

			if os.path.exists('{}/DATA/PRESSURE/{}_{}_{}_{}_PTN.txt'.format(directory, model.lower(), csize, nslice, image)) and ow != 1:
				with file('{}/DATA/PRESSURE/{}_{}_{}_{}_PTN.txt'.format(directory, model.lower(), csize, nslice, image), 'r') as infile:
					PT_temp, PN_temp = np.loadtxt(infile)
			else:
				ZYX = np.rot90(traj.xyz[image])
				zat = ZYX[0] * 10
				yat = ZYX[1] * 10
				xat = ZYX[2] * 10

				PT_temp, PN_temp = PT_PN(Z, nslice, xat, yat, zat, sigma, LJ[0], float(cutoff), DIM)

				with file('{}/DATA/PRESSURE/{}_{}_{}_{}_PTN.txt'.format(directory, model.lower(), csize, nslice, image), 'w') as outfile:
					np.savetxt(outfile, (PT_temp, PN_temp))
		
			tension = integrate.trapz((PN_temp - PT_temp)/(DIM[0] * DIM[1]), Z) * conv
		
			if tension <= 50 and tension >= -50: ST.append(tension)
			else: 
				print 'Anomalous result:', directory, 'Image', image 
				ZYX = np.rot90(traj.xyz[image])
				zat = ZYX[0] * 10
				yat = ZYX[1] * 10
				xat = ZYX[2] * 10
				tensor_tension = (gamma_calc(Z, xat, yat, zat, sigma, LJ[0] * 4.184, float(cutoff), DIM) * 1E26 / constants.N_A)
				PT_temp, PN_temp = PT_PN(Z, xat, yat, zat, sigma, LJ[0], float(cutoff), DIM)
				print tension, tensor_tension
				tension = integrate.trapz((PN_temp - PT_temp)/(DIM[0] * DIM[1]), Z) * conv
				ST.append(tension)
				with file('{}/DATA/PRESSURE/{}_{}_{}_{}_PTN.txt'.format(directory, model.lower(), csize, nslice, image), 'w') as outfile:
					np.savetxt(outfile, (PT_temp, PN_temp))

			TOT_PT += PT_temp / (DIM[0] * DIM[1]) / ntraj
			TOT_PN += PN_temp / (DIM[0] * DIM[1]) / ntraj
			ST1.append(TOTAL_TENSION[(image+offset)*sep-offset])
	
		print np.mean(ST), np.mean(ST1)
		#"""	
		plt.figure(0)
		plt.plot(Z, TOT_PT, linestyle='dashed')
		plt.plot(Z, TOT_PN, linestyle='dashed')
		plt.plot(Z, TOT_PN - TOT_PT)
		plt.figure(1)
		plt.plot(ST)
		plt.plot(ST1)
		#plt.plot(ST2)
		plt.show()
		#"""

