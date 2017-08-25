"""
*************** THERMODYNAMIC MODULE *******************

Analysis of configurational energy, surface tension, surface
energy and surface entropy.

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 29/11/2016 by Frank Longford
"""

import numpy as np
import scipy as sp
import subprocess, time, sys, os, math, copy, gc
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as con
from scipy.optimize import curve_fit, leastsq
import scipy.integrate as spin
from scipy.interpolate import bisplrep, bisplev, splprep, splev

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import utilities as ut
import mdtraj as md


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


def plot_graphs(ENERGY, ENERGY_ERR, TENSION, TENSION_ERR, VAR_TENSION, N_RANGE, A_RANGE, AN_RANGE, Z_RANGE, col, cut):
	"Plots graphs of energy and tension data"

	""" FIGURE PARAMETERS """
	fig_x = 12
	fig_y = 8
	msize = 50
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size='30.0')
	plt.rc('lines', linewidth='2.0', markersize=7)
	plt.rc('axes', labelsize='18.0')
	plt.rc('xtick', labelsize='18.0')
	plt.rc('ytick', labelsize='18.0')

	ZA_RANGE = [np.sqrt(Z_RANGE[x] / A_RANGE[x]) for x in range(len(A_RANGE))]
	EZ_RANGE = [ENERGY[x]  / Z_RANGE[x] for x in range(len(ENERGY))]
	AZ_RANGE = [AN_RANGE[x]  / Z_RANGE[x] for x in range(len(A_RANGE))]

	plt.figure(0, figsize=(fig_x,fig_y))
	plt.scatter(Z_RANGE[cut:], A_RANGE[cut:], color=col, s=msize)
	#plt.scatter(Z_RANGE, intA_RANGE, color=col, s=msize, marker='x')

	plt.figure(1, figsize=(fig_x,fig_y))
	plt.scatter(ZA_RANGE[cut:], TENSION[cut:], color=col, marker='x', s=msize)
	plt.errorbar(ZA_RANGE[cut:], TENSION[cut:], color=col, linestyle='none', yerr=np.array(TENSION_ERR[cut:]))

	plt.figure(2, figsize=(fig_x,fig_y))
	plt.scatter(np.array(ZA_RANGE[cut:]), np.sqrt(np.array(VAR_TENSION[cut:])), color=col, s=msize)
	#plt.scatter(ZintA_RANGE, TENSION_RMS, color=col, s=msize, marker='x')

	plt.figure(5, figsize=(fig_x,fig_y))
	plt.scatter(np.array(AN_RANGE)[cut:], ENERGY[cut:], color=col, s=msize)
	plt.errorbar(np.array(AN_RANGE)[cut:], ENERGY[cut:], color=col, linestyle='none', yerr=np.array(ENERGY_ERR[cut:])*5)

	plt.figure(6, figsize=(fig_x,fig_y))
	plt.scatter(np.array(AN_RANGE)[cut:], TENSION[cut:], color=col, s=msize)
	plt.errorbar(np.array(AN_RANGE)[cut:], TENSION[cut:], color=col, linestyle='none', yerr=np.array(TENSION_ERR[cut:]))


def get_A_excess(root, model, csize, nm, nxy, DIM, nimage, ow_area):

	if os.path.exists('{}/DATA/ENERGY_TENSION/{}_{}_{}_{}_{}_AREA.txt'.format(root, model.lower(), csize, nm, nxy, nimage)) and ow_area.upper() != 'Y':
		with file('{}/DATA/ENERGY_TENSION/{}_{}_{}_{}_{}_AREA.txt'.format(root, model.lower(), csize, nm, nxy, nimage), 'r') as infile:
			tot_a_excess = np.loadtxt(infile)
	else:
		tot_a_excess = []

		for image in xrange(nimage):
			try: 
				_, _, a_excess = ut.curvature_aexcess(root, model, csize, nm, nxy, image)
				tot_a_excess.append(a_excess * DIM[0] * DIM[1])
			except IOError: raise IOError

		with file('{}/DATA/ENERGY_TENSION/{}_{}_{}_{}_{}_AREA.txt'.format(root, model.lower(), csize, nm, nxy, nimage), 'w') as outfile:
			np.savetxt(outfile, tot_a_excess)

	return np.mean(tot_a_excess)


def get_thermo(directory, model, csize, suffix, nslice, ntraj, DIM, nmol, rc, sigma, epsilon, l_constant, ow_ntb):

	energy = 0
	potential = 0
	kinetic = 0
	tension = 0

	lslice = DIM[2] / nslice

	if not os.path.exists("{}/DATA/ENERGY_TENSION".format(directory)): os.mkdir("{}/DATA/ENERGY_TENSION".format(directory))

	FILE = '{}/{}_{}_{}'.format(directory, model.lower(), csize, suffix)
	E, POT, KIN, T_, T_err, ST, TOTAL_ENERGY, TOTAL_POTENTIAL, TOTAL_KINETIC, TOTAL_TENSION, TOTAL_TEMP = ut.read_energy_temp_tension(FILE)

	if rc < 22:

		with file('{}/DATA/DEN/{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, ntraj), 'r') as infile:
			av_density = np.loadtxt(infile)

		corr_e = ut.E_janecek(av_density[-1], rc, sigma, epsilon, lslice, DIM[0]*DIM[1]) 
		corr_st = ut.ST_janecek(av_density[-1], rc, sigma, epsilon, lslice) * 1E26 / con.N_A

		energy += corr_e / nmol
		potential += corr_e / nmol
		tension += corr_st  

		TOTAL_ENERGY = np.array(TOTAL_ENERGY) + corr_e
		TOTAL_POTENTIAL = np.array(TOTAL_POTENTIAL) + corr_e
		TOTAL_TENSION = np.array(TOTAL_TENSION) + corr_st

	energy += E * 4.184 / nmol
	potential += POT * 4.184 / nmol
	kinetic += KIN * 4.184 / nmol
	temp = T_ 
	temp_err = T_err 
	tension += ST 

	ntb = int(len(TOTAL_TENSION) / 100)
	energy_err, potential_err, kinetic_err, tension_err = ut.get_block_error_thermo(TOTAL_ENERGY, TOTAL_POTENTIAL, TOTAL_POTENTIAL, TOTAL_KINETIC, directory, model, csize, ntraj, ntb, ow_ntb)
	energy_err = energy_err / nmol
	potential_err = potential_err / nmol
	kinetic_err = kinetic_err / nmol

	with file('{}/DATA/ENERGY_TENSION/{}_{}_EST.txt'.format(directory, model.lower(), csize), 'w') as outfile:
		np.savetxt(outfile, (energy, energy_err, potential, potential_err, kinetic, kinetic_err, temp, temp_err, tension, tension_err))
	with file('{}/DATA/ENERGY_TENSION/{}_{}_TOTEST.txt'.format(directory, model.lower(), csize), 'w') as outfile:
		np.savetxt(outfile, (TOTAL_ENERGY, TOTAL_POTENTIAL, TOTAL_KINETIC, TOTAL_TENSION, TOTAL_TEMP))


def energy_tension(root, model, suffix, TYPE, folder, sfolder, nfolder, T, rc, LJ, csize, e_constant, l_constant, st_constant, com, ow_area, ow_ntb, ow_est):
	"Get time averaged energy, tension and simulation dimensions"
	
	nbin = 300

	if model.upper() == 'ARGON':
		gam_start = -200 * st_constant
		gam_end = 200 * st_constant
	else:
		gam_start = -500 * st_constant
		gam_end = 500 * st_constant

	ENERGY = np.zeros(nfolder)
	ENERGY_ERR = np.zeros(nfolder)
	TEMP = np.zeros(nfolder)
	TEMP_ERR = np.zeros(nfolder)
	TENSION = np.zeros(nfolder)
	TENSION_ERR = np.zeros(nfolder)
	VAR_TENSION = np.zeros(nfolder)
	N_RANGE = np.zeros(nfolder)
	A_RANGE = np.zeros(nfolder)
	AN_RANGE = np.zeros(nfolder)
	Z_RANGE = np.zeros(nfolder)
	DEN = np.zeros(nfolder)
	OMEGA = np.zeros(nfolder)

	sigma = np.max(LJ[1])
	s_csize = csize

	MARKER = ['o', 'v', 'x', '^', 's', 'p', '+', '*']
	COLOUR = ['b', 'g', 'r', 'c', 'm', 'saddlebrown', 'navy'] 

	for n in range(sfolder, nfolder):
		if TYPE == 'SLAB': directory = root
		else: directory = '{}/{}_{}'.format(root, TYPE.upper(), n)

		print '{}/{}/DATA/parameters.txt'.format(directory, folder.upper())
		if os.path.exists('{}/{}/DATA/parameters.txt'.format(directory, folder.upper())):
			print 'loaded params'
			DIM = np.zeros(3)
			directory = '{}/{}'.format(directory, folder.upper())
			with file('{}/DATA/parameters.txt'.format(directory), 'r') as infile:
				natom, nmol, ntraj, DIM[0], DIM[1], DIM[2] = np.loadtxt(infile)
			natom = int(natom)
			nmol = int(nmol)
			ntraj = int(ntraj)
		else:
			traj = ut.load_nc(directory, folder, model, csize, suffix)				
			directory = '{}/{}'.format(directory, folder.upper())

			natom = int(traj.n_atoms)
			nmol = int(traj.n_residues)
			ntraj = int(traj.n_frames)
			DIM = np.array(traj.unitcell_lengths[0]) * 10

			with file('{}/DATA/parameters.txt'.format(directory), 'w') as outfile:
				np.savetxt(outfile, [natom, nmol, ntraj, DIM[0], DIM[1], DIM[2]])

		lslice = 0.05 * sigma
		nslice = int(DIM[2] / lslice)
		nm = int((DIM[0] + DIM[1]) / (2 * sigma) )
		nxy = int((DIM[0] + DIM[1])/ sigma)

		with file('{}/DATA/DEN/{}_{}_{}_{}_PAR.txt'.format(directory, model.lower(), csize, nslice, ntraj), 'r') as infile:
			param = np.loadtxt(infile)

		print param

		DIM = DIM * l_constant
		A_RANGE[n] = (DIM[0] * DIM[1])
		AN_RANGE[n] = 2 * A_RANGE[n] / nmol
		Z_RANGE[n] = float(param[3]) * 2 * l_constant
		N_RANGE[n] = nmol
		DEN[n] = param[0]
		OMEGA[n] =  param[-1] * 2.1972 * l_constant
		
		"""
		try: A_excess = get_A_excess(directory, model, csize, nm, nxy, DIM , ntraj, ow_area)
		except IOError: pass
		else:
			intA_RANGE[n] = A_excess * l_constant**2
			intAN_RANGE[n] = intA_RANGE[n] / nmol 
                        ZintA_RANGE[n] = np.sqrt(Z_RANGE[n] / intA_RANGE[n])
		"""		

		if not os.path.exists("{}/DATA/ENERGY_TENSION".format(directory)): os.mkdir("{}/DATA/ENERGY_TENSION".format(directory))

		if os.path.exists('{}/DATA/ENERGY_TENSION/{}_{}_EST.txt'.format(directory, model.lower(), csize)) and not ow_est:
			with file('{}/DATA/ENERGY_TENSION/{}_{}_EST.txt'.format(directory, model.lower(), csize), 'r') as infile:
				ENERGY[n], ENERGY_ERR[n], TEMP[n], TEMP_ERR[n], TENSION[n], TENSION_ERR[n] = np.loadtxt(infile)
			with file('{}/DATA/ENERGY_TENSION/{}_{}_TOTEST.txt'.format(directory, model.lower(), csize), 'r') as infile:
				TOTAL_ENERGY, TOTAL_TENSION, TOTAL_TEMP = np.loadtxt(infile)
				print "Data Loaded"

		else:	
			get_thermo(directory, model, csize, suffix, nslice, ntraj, DIM, nmol, rc, sigma, LJ[0], A_RANGE[n], l_constant, lslice, ow_ntb)
			with file('{}/DATA/ENERGY_TENSION/{}_{}_EST.txt'.format(directory, model.lower(), csize), 'r') as infile:
				ENERGY[n], ENERGY_ERR[n], TEMP[n], TEMP_ERR[n], TENSION[n], TENSION_ERR[n] = np.loadtxt(infile)
			with file('{}/DATA/ENERGY_TENSION/{}_{}_TOTEST.txt'.format(directory, model.lower(), csize), 'r') as infile:
				TOTAL_ENERGY, TOTAL_TENSION, TOTAL_TEMP = np.loadtxt(infile)

		print TENSION[n], np.mean(TOTAL_TENSION)

		VAR_TENSION[n] = np.var(TOTAL_TENSION)


		#make_histogram(directory, len(TOTAL_TENSION), TOTAL_TENSION, nbin, gam_start, gam_end, MARKER[n%len(MARKER)], COLOUR[n%len(COLOUR)])

	#plt.show()
	ENERGY = np.array(ENERGY) * e_constant
	ENERGY_ERR = np.array(ENERGY_ERR) * e_constant
	TENSION = np.array(TENSION) * st_constant
	TENSION_ERR = np.array(TENSION_ERR) * st_constant
	VAR_TENSION = np.array(VAR_TENSION) * st_constant**2

	return ENERGY, ENERGY_ERR, TEMP, TEMP_ERR, TENSION, TENSION_ERR, VAR_TENSION, N_RANGE, A_RANGE, AN_RANGE, Z_RANGE, DEN


def make_histogram(root, nsample, gamma, nbin, gam_start, gam_end, mark, col):

	HISTOGRAM = np.zeros(nbin)
	gam_start = np.min(gamma)
	gam_end = np.max(gamma)
	gam_range = gam_end-gam_start
	dgam = gam_range / nbin

	for i in xrange(nsample):
		index = int((gamma[i]-gam_start) * nbin /gam_range)
		try: HISTOGRAM[index] += 1. / (dgam * (nsample-1))
		except IndexError: print index, gamma[i]
	
	X = np.linspace(gam_start, gam_end, nbin)
	param, _ = curve_fit(gaussian_dist, X, HISTOGRAM, [np.mean(gamma), np.std(gamma)**2])
	y_dist = [gaussian_dist(x, param[0], param[1]) for x in X]
	print param, np.mean(gamma), np.std(gamma)**2

	plt.scatter(X, HISTOGRAM, marker=mark, c=col)
	plt.plot(X, y_dist, linestyle='dashed', marker=mark, c=col)

	return HISTOGRAM


def gaussian_dist(x, mean, var): return np.exp(-(x-mean)**2 / (2 * var)) / np.sqrt( 2 * var * np.pi)


def get_U0(model, T, cutoff, csize, sigma, epsilon):

	if model.upper() == 'ARGON': directory = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/W_TEST/W_12'.format(model.upper(), T, cutoff)
	else: directory = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A/W_TEST/W_12'.format(model.upper(), T, cutoff)

	traj = ut.load_nc(directory, 'CUBE', model, csize, 'cube')
	nmol = traj.n_residues
	ntraj = int(traj.n_frames)
	DIM = np.array(traj.unitcell_lengths[0]) * 10				
	directory = '{}/CUBE'.format(directory)

	if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))
	if not os.path.exists("{}/DATA/ENERGY_TENSION".format(directory)): os.mkdir("{}/DATA/ENERGY_TENSION".format(directory))
	
	FILE = '{}/{}_{}_cube'.format(directory, model.lower(), csize)
	E, _, T_, Terr, ST, _, TOTAL_ENERGY, TOTAL_TENSION, _ = ut.read_energy_temp_tension(FILE)
	corr_e = homo_E_correction(float(cutoff), sigma, epsilon, nmol, DIM)

	ntb = int(len(TOTAL_ENERGY) / 100)
	Eerr, _ = ut.get_block_error(directory, model, csize, ntraj, TOTAL_ENERGY, TOTAL_TENSION, ntb, 'N')

	return (np.mean(TOTAL_ENERGY) * 4.184 + corr_e) / nmol, Eerr * 4.184 / nmol


def homo_E_correction(rc, sigma, epsilon, N, DIM): return 8 * np.pi * N**2 * epsilon * sigma**3 / (DIM[0] * DIM[1] * DIM[2]) * (1./9 * (sigma/rc)**9 - 1./3 * (sigma/rc)**3)

def NI_func(Lz, A, pl, rc): return 2 * np.pi * A * pl**2 * rc**3 * (1/3. * Lz - 1/8. * rc)

def std_gamma(Lz, A, pl, rc, omega, std_X): return std_X * np.sqrt(NI_func(Lz, A, pl, rc, omega) / A**2 )

def std_X(Lz, A, pl, rc, omega, std_gamma): return std_gamma / np.sqrt(NI_func(Lz, A, pl, rc, omega) / A**2 )

def main(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, TYPE, folder, sfolder, nfolder, suffix):

	rc = float(cutoff)

	"Conversion of length and surface tension units"
	if model.upper() == 'ARGON':
		LJ[0] = LJ[0] * 4.184
		e_constant = 1 / LJ[0]
                st_constant = ((LJ[1]*1E-10)**2) * con.N_A * 1E-6 / LJ[0]
                l_constant = 1 / LJ[1]
		T = 85
		com = 0
	else: 
		LJ[0] = LJ[0] * 4.184
		e_constant = 1.
                st_constant = 1.
                l_constant = 1E-10
		T = 298
		if model.upper() == 'METHANOL': com = 'COM'
		else: com = 0
	ow_area = bool(raw_input("OVERWRITE INTRINSIC SURFACE AREA? (Y/N): ").upper() == 'Y')
	ow_ntb = bool(raw_input("OVERWRITE SURFACE TENSION ERROR? (Y/N): ").upper() == 'Y')
	ow_est = bool(raw_input("OVERWRITE AVERAGE ENERGY AND TENSION? (Y/N): ").upper() == 'Y')
	(ENERGY, ENERGY_ERR, TEMP, TEMP_ERR, TENSION, TENSION_ERR, VAR_TENSION, N_RANGE, A_RANGE, AN_RANGE, Z_RANGE, DEN) = energy_tension(
		root, model, suffix, TYPE, folder, sfolder, nfolder, T, rc, LJ, csize, e_constant, l_constant, st_constant, com, ow_area, ow_ntb, ow_est)
	
