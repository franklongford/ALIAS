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

def ST_chapela(Zmax, pl, pv, z1, d, r1, r2, sigma, epsilon):

	a, b = spin.quad(lambda r: (-2. * (sigma/r)**13 + (sigma/r)**7) * r**3 * integrate_gamma_s_z(Zmax, pl, pv, z1, d, r), r1, r2)
	return a * 6 * epsilon / sigma * np.pi


def integrate_gamma_s_z(Zmax, pl, pv, z1, d, r):

	a, b = spin.quad(lambda z: integrate_gamma_s(z, pl, pv, z1, d, r), -Zmax/2, Zmax/2)
	return a


def integrate_gamma_s(Z, pl, pv, z1, d, r):

	a, b = spin.quad(lambda s: (1 - 3 * s**2) * ut.den_func(Z+r*s, pl, pv, 0, z1, d) * ut.den_func(Z, pl, pv, 0, z1, d), -1, 1)
	return a 


def E_chapela(DIM, pl, pv, z1, d, r1, r2, sigma, epsilon):

	a, b = spin.quad(lambda r: ((sigma/r)**12 - (sigma/r)**6) * r**2 * integrate_E_z_s(DIM[2], pl, pv, z1, d, r), r1, r2)
	return - a * 2 * epsilon * np.pi * DIM[0] * DIM[1]


def integrate_E_z_s(Zmax, pl, pv, z1, d, r):

	a, b = spin.quad(lambda z: integrate_E_s(z, pl, pv, z1, d, r), -Zmax/2, Zmax/2)
	return a


def integrate_E_s(Z, pl, pv, z1, d, r):

	a, b = spin.quad(lambda s: ut.den_func(Z+r*s, pl, pv, 0, z1, d) * ut.den_func(Z, pl, pv, 0, z1, d), -1, 1)
	return a 


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


def get_thermo(directory, model, csize, suffix, nslice, nframe, DIM, nmol, rc, sigma, epsilon, ow_ntb, corr_meth):
	"Get internal energies in units of mJ mol-1 and surface tension in mJ m-2"

	lslice = DIM[2] / nslice

	FILE = '{}/{}_{}_{}'.format(directory, model.lower(), csize, suffix)
	energy, potential, kinetic, temp, temp_err, tension, TOT_ENERGY, TOT_POTENTIAL, TOT_KINETIC, TOT_TENSION, TOT_TEMP = ut.read_energy_temp_tension(FILE)

	with file('{}/DATA/DEN/{}_{}_{}_DEN.npy'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
		av_density = np.load(infile)

	if corr_meth.upper() == 'C':
		popt, pcov = curve_fit(ut.den_func, np.linspace(0, DIM[2], nslice), av_density[-1], [1., 0., DIM[2]/2., DIM[2]/4., 2.])
		corr_e = E_chapela(DIM, popt[0], popt[1], popt[3], popt[4], float(rc), 200, sigma, epsilon) #total energy correction in kJ mol-1
		corr_st = ST_chapela(DIM[2], popt[0], popt[1], popt[3], popt[4], float(rc), 200, sigma, epsilon) * 1E26 / con.N_A #surface tension correction in mJ m-2
	elif corr_meth.upper() == 'J':
		corr_e = E_janecek(av_density[-1], float(rc), sigma, epsilon, lslice, DIM[0]*DIM[1]) #total energy correction in kJ mol-1
		corr_st = ST_janecek(av_density[-1], float(rc), sigma, epsilon, lslice) * 1E26 / con.N_A #surface tension correction in mJ m-2

	print energy, tension
	print corr_e, corr_st

	energy = (energy * 4.184 + corr_e) / nmol
	potential = (potential * 4.184 + corr_e) / nmol
	kinetic *= 4.184 / nmol
	tension += corr_st  

	TOT_ENERGY = (np.array(TOT_ENERGY) * 4.184 + corr_e) / nmol
	TOT_POTENTIAL = (np.array(TOT_POTENTIAL) * 4.184 + corr_e) / nmol
	TOT_KINETIC = np.array(TOT_KINETIC) * 4.184 / nmol
	TOT_TENSION = np.array(TOT_TENSION) + corr_st

	ntb = int(len(TOT_TENSION) / 100)

	energy_err, potential_err, kinetic_err, tension_err = get_block_error_thermo(TOT_ENERGY, TOT_POTENTIAL, TOT_KINETIC, TOT_TENSION, directory, model, csize, nframe, ntb, ow_ntb)

	with file('{}/DATA/THERMO/{}_{}_E_ST.npy'.format(directory, model.lower(), nframe), 'w') as outfile:
		np.save(outfile, (energy, energy_err, potential, potential_err, kinetic, kinetic_err, temp, temp_err, tension, tension_err))
	with file('{}/DATA/THERMO/{}_{}_TOT_E_ST.npy'.format(directory, model.lower(), nframe), 'w') as outfile:
		np.save(outfile, (TOT_ENERGY, TOT_POTENTIAL, TOT_KINETIC, TOT_TENSION, TOT_TEMP))


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

	if model.upper() == 'ARGON': directory = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/T_TEST'.format(model.upper(), T, cutoff)
	else: directory = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A/T_TEST'.format(model.upper(), T, cutoff)

	traj = ut.load_nc(directory, 'CUBE', model, csize, 'cube')
	nmol = traj.n_residues
	ntraj = int(traj.n_frames)
	DIM = np.array(traj.unitcell_lengths[0]) * 10				
	directory = '{}/CUBE'.format(directory)

	if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))
	if not os.path.exists("{}/DATA/THERMO".format(directory)): os.mkdir("{}/DATA/THERMO".format(directory))
	
	FILE = '{}/{}_{}_cube'.format(directory, model.lower(), csize)
	energy, _, _, _, _, _, TOT_ENERGY, TOT_POTENTIAL, TOT_KINETIC, TOT_TENSION, _ = ut.read_energy_temp_tension(FILE)
	corr_e = homo_E_correction(float(cutoff), sigma, epsilon, nmol, DIM)

	#ntb = int(len(TOT_ENERGY) / 100)
	#energy_err, potential_err, kinetic_err, tension_err = get_block_error_thermo(TOT_ENERGY, TOT_POTENTIAL, TOT_KINETIC, TOT_TENSION, directory, model, csize, ntraj, ntb, False)

	return (energy * 4.184 + corr_e) / nmol


def homo_E_correction(rc, sigma, epsilon, N, DIM): return 8 * np.pi * N**2 * epsilon * sigma**3 / (DIM[0] * DIM[1] * DIM[2]) * (1./9 * (sigma/rc)**9 - 1./3 * (sigma/rc)**3)


def NI_func(Lz, A, pl, rc): return 2 * np.pi * A * pl**2 * rc**3 * (1/3. * Lz - 1/8. * rc)


def std_gamma(Lz, A, pl, rc, omega, std_X): return std_X * np.sqrt(NI_func(Lz, A, pl, rc, omega) / A**2 )


def std_X(Lz, A, pl, rc, omega, std_gamma): return std_gamma / np.sqrt(NI_func(Lz, A, pl, rc, omega) / A**2 )


def get_block_error_thermo(E, POT, KIN, ST, directory, model, csize, ntraj, ntb, ow_ntb):

	if os.path.exists('{}/DATA/THERMO{}_{}_{}_PT.txt'.format(directory, model.lower(), ntraj, ntb)) and not ow_ntb:
		try:
			with file('{}/DATA/THERMO/{}_{}_{}_PT.txt'.format(directory, model.lower(), ntraj, ntb), 'r') as infile:
				pt_e, pt_pot, pt_kin, pt_st = np.loadtxt(infile)
		except: ow_ntb = True
	else: ow_ntb = True

	if ow_ntb: 
		old_pt_st, old_ntb = ut.load_pt('{}/DATA/THERMO'.format(directory), ntraj)

		if old_ntb == 0 or ow_ntb: 
			pt_e, pt_pot, pt_kin, pt_st = ut.block_error((E, POT, KIN, ST), ntb)
		elif old_ntb > ntb:
			with file('{}/DATA/THERMO/{}_{}_{}_PT.txt'.format(directory, model.lower(), ntraj, old_ntb), 'r') as infile:
                        	pt_e, pt_pot, pt_kin, pt_st = np.loadtxt(infile)
			pt_e = pt_e[:ntb]
			pt_pot = pt_pot[:ntb]
			pt_kin = pt_kin[:ntb]
			pt_st = pt_st[:ntb]
		elif old_ntb < ntb:
			with file('{}/DATA/THERMO/{}_{}_{}_PT.txt'.format(directory, model.lower(), ntraj, old_ntb), 'r') as infile:
                        	pt_e, pt_pot, pt_kin, pt_st = np.loadtxt(infile)

			old_pt_e, old_pt_pot, old_pt_kin, old_pt_st = ut.block_error((E, POT, KIN, ST), ntb)

			pt_e = np.concatenate(pt_e, old_pt_e)
			pt_pot = np.concatenate(pt_e_pot, old_pt_pot)
			pt_kin = np.concatenate(pt_e_kin, old_pt_kin)
			pt_st = np.concatenate(pt_st, old_pt_st)
		
		with file('{}/DATA/THERMO/{}_{}_{}_PT.txt'.format(directory, model.lower(), ntraj, ntb), 'w') as outfile:
        		np.savetxt(outfile, (pt_e, pt_pot, pt_kin, pt_st))

        M = len(E)

	corr_time_e = ut.get_corr_time(pt_e, ntb)
	corr_time_pot = ut.get_corr_time(pt_pot, ntb)
	corr_time_kin = ut.get_corr_time(pt_kin, ntb)
	corr_time_st = ut.get_corr_time(pt_st, ntb)

        m_err_e = (np.std(E) * np.sqrt(corr_time_e / M))
	m_err_pot = (np.std(POT) * np.sqrt(corr_time_pot / M))
	m_err_kin = (np.std(KIN) * np.sqrt(corr_time_kin / M))
	m_err_st = (np.std(ST) * np.sqrt(corr_time_st / M))

        return m_err_e, m_err_pot, m_err_kin, m_err_st	
	
