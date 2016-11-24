"""
*************** THERMODYNAMIC MODULE *******************

Analysis of configurational energy, surface tension, surface
energy and surface entropy.

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 24/11/2016 by Frank Longford
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


def plot_graphs(ENERGY, ENERGY_RMS, TENSION, TENSION_RMS, N_RANGE, A_RANGE, intA_RANGE, Z_RANGE, ZA_RANGE, lnZ_RANGE, col):
	"Plots graphs of energy and tension data"

	""" FIGURE PARAMETERS """
	fig_x = 18
	fig_y = 12
	msize = 50
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size='20.0')
	plt.rc('lines', linewidth='2.0', markersize=7)
	plt.rc('axes', labelsize='18.0')
	plt.rc('xtick', labelsize='18.0')
	plt.rc('ytick', labelsize='18.0')

	plt.figure(0, figsize=(fig_x,fig_y))
	plt.scatter(Z_RANGE, intA_RANGE, color=col, s=msize)

	plt.figure(1, figsize=(fig_x,fig_y))
	plt.scatter(ZA_RANGE, TENSION, color=col, marker='x', s=msize)
	plt.errorbar(ZA_RANGE, TENSION, color=col, linestyle='none', yerr=TENSION_RMS)

	plt.figure(2, figsize=(fig_x,fig_y))
	plt.scatter(ZA_RANGE, TENSION_RMS, color=col, s=msize)

	plt.figure(3, figsize=(fig_x,fig_y))
	plt.scatter(1./intA_RANGE, TENSION, color=col, s=msize)

	plt.figure(4, figsize=(fig_x,fig_y))
	plt.scatter(lnZ_RANGE, TENSION, color=col, s=msize)

	plt.figure(5, figsize=(fig_x,fig_y))
	plt.scatter(1./Z_RANGE, TENSION, color=col, s=msize)


def get_A_excess(root, model, csize, nm, nxy, DIM, nimage):

	ow_area = 'Y'

	if os.path.exists('{}/DATA/ENERGY_TENSION/{}_{}_{}_{}_{}_AREA.txt'.format(root, model.lower(), csize, nm, nxy, nimage)) and ow_area != 'Y':
		with file('{}/DATA/ENERGY_TENSION/{}_{}_{}_{}_{}_AREA.txt'.format(root, model.lower(), csize, nm, nxy, nimage), 'r') as infile:
			tot_a_excess = np.loadtxt(infile)
	else:
		tot_a_excess = []

		for image in xrange(nimage):
			if os.path.exists('{}/DATA/INTDEN/{}_{}_{}_{}_{}_CURVE.npz'.format(root, model.lower(), csize, nm, nxy, image)):
				with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_CURVE.npz'.format(root, model.lower(), csize, nm, nxy, image), 'r') as infile:
					npzfile = np.load(infile)
					DX1 = npzfile['DX1']
					DY1 = npzfile['DY1']
					DX2 = npzfile['DX2']
					DY2 = npzfile['DY2']
				a_excess = 0
				for i in xrange(nxy):
					for j in xrange(nxy):
						dS1 = np.sqrt(DX1[i][j]**2 + DY1[i][j]**2 + 1)
						dS2 = np.sqrt(DX2[i][j]**2 + DY2[i][j]**2 + 1)
						a_excess += DIM[0] * DIM[1] * ( (dS1 + dS2) / nxy**2) 
	
				tot_a_excess.append(a_excess)

		with file('{}/DATA/ENERGY_TENSION/{}_{}_{}_{}_{}_AREA.txt'.format(root, model.lower(), csize, nm, nxy, nimage), 'w') as outfile:
			np.savetxt(outfile, tot_a_excess)

	return np.mean(tot_a_excess)


def energy_tension(root, model, suffix, TYPE, folder, nfolder, T, rc, LJ, csize, l_constant, st_constant, ntraj, com, ow_all):
	"Get time averaged energy, tension and simulation dimensions"

	ENERGY = np.zeros(nfolder)
	ENERGY_ERR = np.zeros(nfolder)
	TEMP = np.zeros(nfolder)
	TEMP_ERR = np.zeros(nfolder)
	TENSION = np.zeros(nfolder)
	TENSION_ERR = np.zeros(nfolder)
	N_RANGE = np.zeros(nfolder)
	A_RANGE = np.zeros(nfolder)
	intA_RANGE = np.zeros(nfolder)
	Z_RANGE = np.zeros(nfolder)
	ZA_RANGE = np.zeros(nfolder)
	lnZ_RANGE = np.zeros(nfolder)

	sigma = np.max(LJ[1])

	for n in range(nfolder):
		directory = '{}/{}_{}'.format(root, TYPE.upper(), n)
		if not os.path.exists('{}/{}/{}_{}_{}_{}.nc'.format(directory, folder.upper(), model.lower(), csize, suffix, 800)): 
			ut.make_nc(directory, folder.upper(),  model.lower(), csize, suffix, ntraj, 'N')
		traj = ut.load_nc(directory, folder.upper())							
		directory = '{}/{}'.format(directory, folder.upper())

		natom = traj.n_atoms
		nmol = traj.n_residues
		DIM = np.array(traj.unitcell_lengths[0]) * 10
		lslice = 0.05 * sigma
		nslice = int(DIM[2] / lslice)
		nm = int((DIM[0] + DIM[1]) / (2 * sigma) )
		nxy = 30

		with file('{}/DATA/DEN/{}_{}_{}_{}_PAR.txt'.format(directory, model.lower(), csize, nslice, ntraj), 'r') as infile:
			param = np.loadtxt(infile)

		DIM = DIM * l_constant
		A_RANGE[n] = (2 * DIM[0] * DIM[1]) / l_constant**2
		intA_RANGE[n] = get_A_excess(directory, model, csize, nm, nxy, DIM, ntraj) / l_constant**2
		Z_RANGE[n] = float(param[3]) / l_constant

		ZA_RANGE[n] = np.sqrt(Z_RANGE[n] / intA_RANGE[n])
		lnZ_RANGE[n] = np.log(Z_RANGE[n]) / intA_RANGE[n]
		N_RANGE[n] = nmol

		if not os.path.exists("{}/DATA/ENERGY_TENSION".format(directory)): os.mkdir("{}/DATA/ENERGY_TENSION".format(directory))
		
		if os.path.exists('{}/DATA/ENERGY_TENSION/{}_{}_{}_TOTEST.txt'.format(directory, model.lower(), csize, ntraj)) and ow_all != 'Y':
			with file('{}/DATA/ENERGY_TENSION/{}_{}_{}_EST.txt'.format(directory, model.lower(), csize, ntraj), 'r') as infile:
				try: 
					ENERGY[n], ENERGY_ERR[n], TEMP[n], TEMP_ERR[n], TENSION[n], TENSION_ERR[n] = np.loadtxt(infile)
					MAKE = 'N'
				except ValueError: MAKE = 'Y'
			with file('{}/DATA/ENERGY_TENSION/{}_{}_{}_TOTEST.txt'.format(directory, model.lower(), csize, ntraj), 'r') as infile:
				TOTAL_ENERGY, TOTAL_TENSION = np.loadtxt(infile)

		else: MAKE = 'Y'

		if MAKE == 'Y':
			TOTAL_ENERGY = []
			TOTAL_TENSION = []

			with file('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, ntraj), 'r') as infile:
				av_mass_den, _, av_mol_den, _ = np.loadtxt(infile)

			corr_e = ut.E_janecek(av_mol_den, rc, sigma, LJ[0], lslice, A_RANGE[n]) * 1E-6 * con.N_A
			corr_st = ut.ST_janecek(av_mol_den, rc, sigma, LJ[0], lslice) * 1E20
			ENERGY[n] += corr_e / nmol
			TENSION[n] += corr_st

			for image in xrange(ntraj):
				sys.stdout.write("PROCESSING {} IMAGE {} \r".format(directory, image) )
				sys.stdout.flush()

				FILE = '{}/{}_{}_{}{}'.format(directory, model.lower(), csize, suffix, image)
				E, _, T_, Terr, ST, _, energy, tension = ut.read_energy_temp_tension(FILE)

				TOTAL_ENERGY += energy
				TOTAL_TENSION += tension

				ENERGY[n] += E * 4.184 / (nmol * ntraj)
				TEMP[n] += T_ / ntraj
				TEMP_ERR[n] += Terr / ntraj
				TENSION[n] += ST * st_constant / ntraj

			TOTAL_ENERGY = np.array(TOTAL_ENERGY) * 4.184 + corr_e
			TOTAL_TENSION = (np.array(TOTAL_TENSION) + corr_st) * st_constant
			
			ENERGY_ERR[n] = np.std(TOTAL_ENERGY) / np.sqrt(len(TOTAL_ENERGY))#ut.block_error(TOTAL_ENERGY, ntb)
			TENSION_ERR[n] = np.std(TOTAL_TENSION)/ np.sqrt(len(TOTAL_TENSION))#ut.block_error(TOTAL_TENSION, ntb)
			
			with file('{}/DATA/ENERGY_TENSION/{}_{}_{}_EST.txt'.format(directory, model.lower(), csize, ntraj), 'w') as outfile:
				np.savetxt(outfile, (ENERGY[n], ENERGY_ERR[n], TEMP[n], TEMP_ERR[n], TENSION[n], TENSION_ERR[n]))
			with file('{}/DATA/ENERGY_TENSION/{}_{}_{}_TOTEST.txt'.format(directory, model.lower(), csize, ntraj), 'w') as outfile:
				np.savetxt(outfile, (TOTAL_ENERGY, TOTAL_TENSION))

	return ENERGY, ENERGY_ERR, TEMP, TEMP_ERR, TENSION, TENSION_ERR,N_RANGE, A_RANGE, intA_RANGE, Z_RANGE, ZA_RANGE, lnZ_RANGE


def make_histogram(root, nimage, gamma, nbin, gam_start, gam_end):

	HISTOGRAM = np.zeros(nbins)

	for image in xrange(nimage):
		index = int((gamma[i]+(gam_end-gam_start)/2.) * nbin / (gam_end-gam_start))
		try: HISTOGRAM[index] += 1. / (len(gamma) * nimage)
		except IndexError: print index, gamma[i]
	
	return HISTOGRAM


def main(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, TYPE, folder, nfolder, suffix, ntraj):

	groot = "/home/fl7g13/Documents/Figures/Surface"

	TOT_ENERGY = []
	TOT_TEMP = []
	TOT_AN = []
	TOT_ZA_RANGE = []
	TOT_lnZ_RANGE = []
	TOT_TENSION = []
	TOT_TENSION_ERR = []

	nbin = 300
	rc = float(cutoff)

	"Conversion of length and surface tension units"
	if model.upper() == 'ARGON':
		LJ[0] = LJ[0] * 4.184E6 / con.N_A
		st_constant = ((LJ[1]*1E-10)**2) / LJ[0]
		l_constant = 1 / LJ[1]
		T = 85
		gam_start = -300 * st_constant
		gam_end = 300 * st_constant
		com = 0
	else: 
		LJ[0] = LJ[0] * 4.184E6 / con.N_A		
		st_constant = 1.
		l_constant = 1E-10
		T = 298
		gam_start = -500 * st_constant
		gam_end = 500 * st_constant
		if model.upper() == 'METHANOL': com = 'COM'
		else: com = 0

	nfolder = 12
	ENERGY, ENERGY_ERR, TEMP, TEMP_ERR, TENSION, TENSION_ERR, N_RANGE, A_RANGE, intA_RANGE, Z_RANGE, ZA_RANGE, lnZ_RANGE = energy_tension(
		root, model, suffix, TYPE, folder, nfolder, T, rc, LJ, csize, l_constant, st_constant, ntraj, com, 'Y')
	plot_graphs(ENERGY, ENERGY_ERR, TENSION, TENSION_ERR, N_RANGE, A_RANGE, intA_RANGE, Z_RANGE, ZA_RANGE, lnZ_RANGE, 'b')

	for n in xrange(nfolder):
		TOT_ENERGY.append(ENERGY[n])
		TOT_TEMP.append(TEMP[n])
		TOT_ZA_RANGE.append(ZA_RANGE[n])
		TOT_lnZ_RANGE.append(lnZ_RANGE[n])
		TOT_TENSION.append(TENSION[n])
		TOT_TENSION_ERR.append(TENSION_ERR[n])
		TOT_AN.append(A_RANGE[n] / N_RANGE[n])

	""" FIGURE PARAMETERS """
	fig_x = 18
	fig_y = 12
	msize = 50
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size='20.0')
	plt.rc('lines', linewidth='2.0', markersize=7)
	plt.rc('axes', labelsize='18.0')
	plt.rc('xtick', labelsize='18.0')
	plt.rc('ytick', labelsize='18.0')

	m, c, r_value, p_value, std_err = stats.linregress(TOT_ZA_RANGE, TOT_TENSION_ERR)
	print "Average TENSION RMS Lz/A =", m , c, r_value
	y_data3 = [m * x + c for x in TOT_ZA_RANGE]

	plt.figure(0, figsize=(fig_x,fig_y))
	plt.xlabel(r'\LARGE{$L_z$ (\AA)}')
	plt.ylabel(r'\LARGE{$A$ (\AA$^{-2}$)}')

	plt.figure(1, figsize=(fig_x,fig_y))
	plt.xlabel(r'\LARGE{$\sqrt{L_z/A}$ (m$^{-\frac{1}{2}}$)}')
	if model.upper() == 'ARGON': plt.ylabel(r'\LARGE{$\bar{\gamma}$}')
	else: plt.ylabel(r'\LARGE{$\bar{\gamma}$ (mJ m$^{-2}$)}')
	plt.savefig('/home/fl7g13/Documents/Figures/Surface/{}_surface_tension_paper.pdf'.format(model.lower()))

	plt.figure(2, figsize=(fig_x,fig_y))
	plt.plot(TOT_ZA_RANGE, y_data3, linestyle='dashed', color='black')
	plt.xlabel(r'\LARGE{$\sqrt{L_z/A}$ (m$^{-\frac{1}{2}}$)}')
	if model.upper() == 'ARGON': plt.ylabel(r'\LARGE{$\sigma\bar{\gamma}$}')
	else: plt.ylabel(r'\LARGE{$\sigma\bar{\gamma}$ (mJ m$^{-2}$)}')
	plt.savefig('/home/fl7g13/Documents/Figures/Surface/{}_surface_tension_rms_paper.pdf'.format(model.lower()))

	plt.figure(3, figsize=(fig_x,fig_y))
	plt.xlabel(r'\LARGE{$1/A$ (\AA$^{-2}$)}')
	if model.upper() == 'ARGON': plt.ylabel(r'\LARGE{$\bar{\gamma}$}')
	else: plt.ylabel(r'\LARGE{$\bar{\gamma}$ (mJ m$^{-2}$)}')
	plt.savefig('/home/fl7g13/Documents/Figures/Surface/{}_invA_vs_surface_tension_paper.pdf'.format(model.lower()))

	plt.figure(4, figsize=(fig_x,fig_y))
	plt.xlabel(r'\LARGE{$\ln(L_z)/A$ (\AA$^{-2}$)}')
	if model.upper() == 'ARGON': plt.ylabel(r'\LARGE{$\bar{\gamma}$}')
	else: plt.ylabel(r'\LARGE{$\bar{\gamma}$ (mJ m$^{-2}$)}')
	plt.savefig('/home/fl7g13/Documents/Figures/Surface/{}_lnLz_vs_surface_tension_paper.pdf'.format(model.lower()))

	plt.figure(5, figsize=(fig_x,fig_y))
	plt.xlabel(r'\LARGE{$1/L_z$ (\AA$^{-1}$)}')
	if model.upper() == 'ARGON': plt.ylabel(r'\LARGE{$\bar{\gamma}$}')
	else: plt.ylabel(r'\LARGE{$\bar{\gamma}$ (mJ m$^{-2}$)}')
	plt.savefig('/home/fl7g13/Documents/Figures/Surface/{}_invLz_vs_surface_tension_paper.pdf'.format(model.lower()))

	XAN = np.array(TOT_AN) * 1E-20 * con.N_A
	YEN = np.array(TOT_ENERGY)
	m, c, r_value, p_value, std_err = stats.linregress( XAN , YEN)
	print "Average Surface ENERGY {} =".format(model.upper()), m*1E6 , c, r_value

	tot_gamma = np.array(TOT_TENSION)
	gamma_err = np.sum([ 1. / TOT_TENSION_ERR[n]**2 for n in xrange(len(TOT_TENSION_ERR))])
	av_gamma = np.sum([ tot_gamma[n] / (TOT_TENSION_ERR[n]**2 * gamma_err) for n in xrange(len(TOT_TENSION_ERR))]) / st_constant
	print "Average Surface TENSION {} = {}".format(model.upper(), av_gamma)
	ydata = map(lambda x: x * m + c, np.linspace(np.min(XAN), np.max(XAN), len(XAN)))
	print "Average Surface ENTROPY {} = {}".format(model.upper(), (m*1E6 - av_gamma) / np.mean(TOT_TEMP) )

	plt.figure(20, figsize=(fig_x,fig_y))
	plt.plot(np.linspace(np.min(XAN), np.max(XAN), len(XAN)), ydata, linestyle='dashed', color='black')
	plt.scatter(np.array(TOT_AN) * 1E-20 * con.N_A, TOT_ENERGY, color='b', s=msize)
	plt.xlabel(r'\LARGE{$A/N$ (m$^{2}$ mol$^{-1}$)}')
	plt.ylabel(r'\LARGE{$U/N$ (mJ mol$^{-1}$)}')
	plt.savefig('/home/fl7g13/Documents/Figures/Surface/{}_surface_energy_paper.pdf'.format(model.lower()))
	plt.show()
