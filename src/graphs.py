"""
*************** POSITION ANALYSIS *******************

PROGRAM INPUT:
	       

PROGRAM OUTPUT: 
		
		

***************************************************************
Created 11/05/16 by Frank Longford

Last modified 11/05/16 by Frank Longford
"""

import numpy as np
import scipy as sp
import subprocess, time, sys, os, math, copy
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
import intrinsic_sampling_method as ism
import thermodynamics as thermo

""" FIGURE PARAMETERS """
fig_x = 12
fig_y = 8
msize = 50
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='20.0')
plt.rc('lines', linewidth='2.0', markersize=7)
plt.rc('axes', labelsize='25.0')
plt.rc('xtick', labelsize='25.0')
plt.rc('ytick', labelsize='25.0')

def linear(x, m, c): return m*x + c

def print_graphs_density(directory, model, nsite, AT, M, nslice, cutoff, csize, folder, suffix, nframe, DIM):

	groot = "/home/fl7g13/Documents/Thesis/Figures/{}_{}_{}".format(model.upper(), csize, cutoff)
	if not os.path.exists(groot): os.mkdir(groot)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	with file('{}/DATA/DEN/{}_{}_{}_PAR.txt'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
		param = np.loadtxt(infile)

	Z3 = np.linspace(-DIM[2]/2+param[3], DIM[2]/2+param[3], nslice)
	
	with file('{}/DATA/DEN/{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
		av_density = np.loadtxt(infile)

	mass_den = av_density[0]
	mol_den = av_density[-1]

	sm_mol_den = [ut.den_func(z, param[0], param[1], param[2], param[3], param[4]) for z in Z1]

	m_size = 40

	plt.figure(0, figsize=(fig_x,fig_y))
	plt.scatter(Z3, mass_den, c='green', lw=0, s=m_size)
	plt.plot(Z3, sm_mol_den, c='green')
	plt.xlabel(r'$z$ (\AA)')
	plt.ylabel(r'$\rho(z)$ (g cm$^{-3}$)')
	plt.axis([-10., 15, 0, np.max(mass_den) + 0.1 * np.max(mass_den)])
	#plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_mass_density.png'.format(groot, model.lower(), nslice, nframe))
	plt.savefig('{}/{}_{}_{}_mass_density.pdf'.format(groot, model.lower(), nslice, nframe))

	plt.figure(1, figsize=(fig_x,fig_y))
	plt.scatter(Z3, mol_den, c='green', lw=0, s=m_size)
	plt.xlabel(r'$z$ (\AA)')
	plt.ylabel(r'$\rho(z)$ (\AA$^{-3}$)')
	plt.axis([-10., 15, 0, 0.04])
	#plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_mol_density.png'.format(groot, model.lower(), nslice, nframe))
	plt.savefig('{}/{}_{}_{}_mol_density.pdf'.format(groot, model.lower(), nslice, nframe))

	plt.figure(2, figsize=(fig_x,fig_y))
	for at_type in atom_types: plt.plot(Z3, av_density[1 + atom_types.index(at_type)])
	plt.xlabel(r'$z$ (\AA)')
	plt.ylabel(r'$\rho(z)$ (\AA$^{-3}$)')
	plt.axis([-10., 15, 0, 0.080])
	#plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_atom_density.png'.format(groot, model.lower(), nslice, nframe))
	plt.savefig('{}/{}_{}_{}_atom_density.pdf'.format(groot, model.lower(), nslice, nframe))

	plt.show()

def print_graphs_intrinsic_density(directory, model, nsite, AT, M, nslice, nm, QM, n0, phi, cutoff, csize, folder, suffix, nframe, DIM):

	groot = "/home/fl7g13/Documents/Thesis/Figures/{}_{}_{}".format(model.upper(), csize, cutoff)
	if not os.path.exists(groot): os.mkdir(groot)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	with file('{}/DATA/DEN/{}_{}_{}_PAR.txt'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
		param = np.loadtxt(infile)

	Z3 = np.linspace(-DIM[2]/2+param[3], DIM[2]/2+param[3], nslice)

	for qm in QM:

		with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), nframe), 'r') as infile:
			int_av_density = np.loadtxt(infile)

		mol_int_den = 0.5 * (int_av_density[-4] + int_av_density[-3][::-1])
		mass_int_den = mol_int_den * np.sum(M) / con.N_A * 1E24

		m_size = 40

		plt.figure(3, figsize=(fig_x,fig_y))
		plt.plot(Z2, mol_int_den)
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$\rho(z)$ (\AA$^{-3}$)')
		plt.axis([-10., 15, 0, 0.10])
		#plt.legend(loc=3)
		plt.savefig('{}/{}_{}_{}_{}_{}_int_mol_density.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
		plt.savefig('{}/{}_{}_{}_{}_{}_int_mol_density.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))
		
		"""
		plt.figure(4, figsize=(fig_x,fig_y))
		plt.plot(Z2, mass_int_den)
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$\rho(z)$ (g cm$^{-3}$)')
		plt.axis([-10., 15, 0, 3.0])
		#plt.legend(loc=3)
		plt.savefig('{}/{}_{}_{}_{}_{}_int_mass_density.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
		plt.savefig('{}/{}_{}_{}_{}_{}_int_mass_density.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))
		"""

		plt.figure(5, figsize=(fig_x,fig_y))
		for at_type in atom_types: plt.plot(Z2, int_av_density[1 + atom_types.index(at_type)])
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$\rho(z)$ (\AA$^{-3}$)')
		plt.axis([-10., 15, 0, 0.15])
		#plt.legend(loc=4)
		plt.savefig('{}/{}_{}_{}_{}_{}_int_atom_density.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
		plt.savefig('{}/{}_{}_{}_{}_{}_int_atom_density.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))

		plt.figure(6, figsize=(fig_x,fig_y))
		plt.plot(Z2, int_av_density[-4])
		#plt.plot(np.linspace(0, max_r, rslice), rad_density_array[0], c='green', linestyle='dotted')
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$g(z)$')
		plt.axis([-10., 15, 0, 0.08])
		#plt.legend(loc=3)
		plt.savefig('{}/{}_{}_{}_{}_{}_int_mol_density_1.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
		plt.savefig('{}/{}_{}_{}_{}_{}_int_mol_density_1.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))

		plt.figure(7, figsize=(fig_x,fig_y))
		plt.plot(Z2, int_av_density[-3])
		#plt.plot(np.linspace(0, max_r, rslice), rad_density_array[0], c='green', linestyle='dotted')
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$g(z)$')
		plt.axis([-10., 15, 0, 0.08])
		#plt.legend(loc=3)
		plt.savefig('{}/{}_{}_{}_{}_{}_int_mol_density_2.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
		plt.savefig('{}/{}_{}_{}_{}_{}_int_mol_density_2.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))	
		
		plt.figure(8, figsize=(fig_x,fig_y))
		plt.plot(Z3, int_av_density[-2])
		#plt.plot(np.linspace(0, max_r, rslice), rad_density_array[0], c='green', linestyle='dotted')
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$g(z)$')
		plt.axis([-10., 15, 0, 0.08])
		#plt.legend(loc=3)
		plt.savefig('{}/{}_{}_{}_{}_{}_eff_mol_density1.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
		plt.savefig('{}/{}_{}_{}_{}_{}_eff_mol_density1.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))
	
		plt.figure(9, figsize=(fig_x,fig_y))
		plt.plot(Z3, int_av_density[-1])
		#plt.plot(np.linspace(0, max_r, rslice), rad_density_array[0], c='green', linestyle='dotted')
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$g(z)$')
		plt.axis([15., 30, 0, 0.08])
		#plt.legend(loc=3)
		plt.savefig('{}/{}_{}_{}_{}_{}_eff_mol_density2.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
		plt.savefig('{}/{}_{}_{}_{}_{}_eff_mol_density2.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))

	plt.close('all')

def print_graphs_orientational(directory, model, nsite, AT, nslice, a_type, cutoff, csize, folder, suffix, nframe, DIM):

	groot = "/home/fl7g13/Documents/Thesis/Figures/{}_{}_{}".format(model.upper(), csize, cutoff)
	if not os.path.exists(groot): os.mkdir(groot)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	with file('{}/DATA/DEN/{}_{}_{}_PAR.txt'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
		param = np.loadtxt(infile)
	with file('{}/DATA/EULER/{}_{}_{}_{}_EUL.txt'.format(directory, model.lower(), a_type, nslice, nframe), 'r') as infile:
		axx, azz, av_theta, av_phi, av_varphi, P1, P2 = np.loadtxt(infile)

	Z3 = np.linspace(-DIM[2]/2+param[3], DIM[2]/2+param[3], nslice)
	
	a = ut.get_polar_constants(model, a_type)
	av_a = np.mean(a)
	d_a = 0.04 * av_a

	m_size = 40

	plt.figure(10, figsize=(fig_x,fig_y))
        plt.scatter(Z2, av_theta, c='blue', lw=0, s=m_size)
	plt.scatter(Z2, av_phi, c='red', lw=0, s=m_size)
        plt.xlabel(r'$z$ (\AA)')
        plt.ylabel(r'$\left<\theta\right>(z)$')
        #plt.axis([-DIM[2]/2., DIM[2]/2., 1.44, 1.5])
        plt.legend(loc=3)
        plt.savefig('{}/{}_{}_{}_theta.png'.format(groot, model.lower(), nslice, nframe))

	plt.figure(11, figsize=(fig_x,fig_y))
	plt.scatter(Z3, axx, c='red', label=r'$\alpha_\parallel$', lw=0, s=m_size)
	plt.scatter(Z3, azz, c='blue', label=r'$\alpha_\perp$', lw=0, s=m_size)
	plt.xlabel(r'$z$ (\AA)')
	plt.ylabel(r'$\alpha(z)$ (\AA$^{3}$)')
	#plt.axis([-DIM[2]/2., DIM[2]/2., av_a - 0.05, av_a + 0.05])
	plt.axis([-10., +15, av_a - d_a, av_a + d_a])
	plt.legend(loc=4)
	plt.savefig('{}/{}_{}_{}_polarisability.png'.format(groot, model.lower(), nslice, nframe))
	plt.savefig('{}/{}_{}_{}_polarisability.pdf'.format(groot, model.lower(), nslice, nframe))

	plt.close('all')

def print_graphs_intrinsic_orientational(directory, model, nsite, AT, nslice, nm, QM, n0, phi, a_type, cutoff, csize, folder, suffix, nframe, DIM):

	groot = "/home/fl7g13/Documents/Thesis/Figures/{}_{}_{}".format(model.upper(), csize, cutoff)
	if not os.path.exists(groot): os.mkdir(groot)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	with file('{}/DATA/DEN/{}_{}_{}_PAR.txt'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
		param = np.loadtxt(infile)
	
	Z3 = np.linspace(-DIM[2]/2+param[3], DIM[2]/2+param[3], nslice)
	
	m_size = 40

	a = ut.get_polar_constants(model, a_type)
	av_a = np.mean(a)
	d_a = 0.04 * av_a

	for qm in QM:
		with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_{}_{}_EUL1.txt'.format(directory, model.lower(), a_type, nslice, nm, qm, n0, int(1/phi + 0.5), nframe), 'r') as infile:
			int_axx1, int_azz1, int_av_theta1, int_av_phi1, int_av_varphi1, int_P11, int_P21 = np.loadtxt(infile)
		with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_{}_{}_EUL2.txt'.format(directory, model.lower(), a_type, nslice, nm, qm, n0, int(1/phi + 0.5), nframe), 'r') as infile:
			int_axx2, int_azz2, int_av_theta2, int_av_phi2, int_av_varphi2, int_P12, int_P22 = np.loadtxt(infile)

		av_int_axx = (int_axx1 + int_axx2[::-1]) / 2.
		av_int_azz = (int_azz1 + int_azz2[::-1]) / 2.

		plt.figure(12, figsize=(fig_x,fig_y))
		plt.scatter(Z2, int_av_theta1, c='blue', lw=0, s=m_size)
		plt.scatter(Z2, int_av_phi1, c='red', lw=0, s=m_size)
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$\left<\theta\right>(z)$')
		#plt.axis([-6., 8, 1.44, 1.5])
		plt.legend(loc=3)

		plt.figure(13, figsize=(fig_x,fig_y))
		plt.scatter(Z2, int_av_theta2, c='blue', lw=0, s=m_size)
		plt.scatter(Z2, int_av_phi2, c='red', lw=0, s=m_size)
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$\left<\theta\right>(z)$')
		#plt.axis([-6., 8, 1.44, 1.5])
		plt.legend(loc=3)

		plt.figure(14, figsize=(fig_x,fig_y))
		plt.scatter(Z2, int_axx1, c='red', label=r'$\alpha_\parallel$', lw=0, s=m_size)
		plt.scatter(Z2, int_azz1, c='blue', label=r'$\alpha_\perp$', lw=0, s=m_size)
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$\alpha(z)$ (\AA$^{3}$)')
		plt.axis([-10, 15, av_a - d_a, av_a + d_a])
		plt.legend(loc=4)

		plt.figure(15, figsize=(fig_x,fig_y))
		plt.scatter(Z2, int_axx2, c='red', label=r'$\alpha_\parallel$', lw=0, s=m_size)
		plt.scatter(Z2, int_azz2, c='blue', label=r'$\alpha_\perp$', lw=0, s=m_size)
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$\alpha(z)$ (\AA$^{3}$)')
		plt.axis([-10, 15, av_a - d_a, av_a + d_a])
		plt.legend(loc=3)

		plt.figure(16, figsize=(fig_x,fig_y))
		plt.scatter(Z2, av_int_axx, c='red', label=r'$\alpha_\parallel$', lw=0, s=m_size)
		plt.scatter(Z2, av_int_azz, c='blue', label=r'$\alpha_\perp$', lw=0, s=m_size)
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$\alpha(z)$ (\AA$^{3}$)')
		plt.axis([-10, 15, av_a - d_a, av_a + d_a])
		plt.legend(loc=4)
		
	plt.figure(12, figsize=(fig_x,fig_y))
	plt.savefig('{}/{}_{}_theta1.png'.format(groot, model.lower(), cutoff))
	plt.figure(13, figsize=(fig_x,fig_y))
	plt.savefig('{}/{}_{}_theta2.png'.format(groot, model.lower(), cutoff))
	plt.figure(14, figsize=(fig_x,fig_y))
	plt.savefig('{}/{}_{}_{}_{}_{}_int_polarisability_1.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
	plt.savefig('{}/{}_{}_{}_{}_{}_int_polarisability_1.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))
	plt.figure(15, figsize=(fig_x,fig_y))
	plt.savefig('{}/{}_{}_{}_{}_{}_int_polarisability_2.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
	plt.savefig('{}/{}_{}_{}_{}_{}_int_polarisability_2.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))
	plt.figure(16, figsize=(fig_x,fig_y))
	plt.savefig('{}/{}_{}_{}_{}_{}_int_polarisability.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
	plt.savefig('{}/{}_{}_{}_{}_{}_int_polarisability.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))

	plt.close('all')

def print_graphs_dielectric(directory, model, nsite, AT, nslice, a_type, cutoff, csize, folder, suffix, nframe, DIM):

	groot = "/home/fl7g13/Documents/Thesis/Figures/{}_{}_{}".format(model.upper(), csize, cutoff)
	if not os.path.exists(groot): os.mkdir(groot)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	with file('{}/DATA/DEN/{}_{}_{}_PAR.txt'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
		param = np.loadtxt(infile)
	with file('{}/DATA/DIELEC/{}_{}_{}_{}_DIE.txt'.format(directory, model.lower(), a_type, nslice, nframe), 'r') as infile:
		exx, ezz = np.loadtxt(infile)

	Z3 = np.linspace(-DIM[2]/2+param[3], DIM[2]/2+param[3], nslice)

	plt.figure(17, figsize=(fig_x,fig_y))
	plt.scatter(Z3, exx, c='orange', label=r'$\epsilon_\parallel$', lw=0, s=msize, edgecolors=None)
	plt.scatter(Z3, ezz, c='brown', label=r'$\epsilon_\perp$', lw=1, s=msize, marker='x')
	plt.xlabel(r'$z$ (\AA)')
	plt.ylabel(r'$\epsilon(z)$ (a.u)')
	plt.axis([-10., +15, 1, 2.0])
	plt.legend(loc=4)
	plt.savefig('{}/{}_{}_{}_dielectric.png'.format(groot, model.lower(), nslice, nframe))
	plt.savefig('{}/{}_{}_{}_dielectric.pdf'.format(groot, model.lower(), nslice, nframe))

	plt.close('all')


def print_graphs_intrinsic_dielectric(directory, model, nsite, AT, nslice, nm, QM, n0, phi, a_type, cutoff, csize, folder, suffix, nframe, DIM):

	groot = "/home/fl7g13/Documents/Thesis/Figures/{}_{}_{}".format(model.upper(), csize, cutoff)
	if not os.path.exists(groot): os.mkdir(groot)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	with file('{}/DATA/DEN/{}_{}_{}_PAR.txt'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
		param = np.loadtxt(infile)
	
	Z3 = np.linspace(-DIM[2]/2+param[3], DIM[2]/2+param[3], nslice)

	for qm in QM:

		with file('{}/DATA/INTDIELEC/{}_{}_{}_{}_{}_{}_{}_{}_DIE.txt'.format(directory, model.lower(), a_type, nslice, nm, qm, n0, int(1./phi + 0.5), nframe), 'r') as infile:
			int_exx, int_ezz = np.loadtxt(infile)
		with file('{}/DATA/INTDIELEC/{}_{}_{}_{}_{}_{}_{}_{}_CWDIE.txt'.format(directory, model.lower(), a_type, nslice,nm, qm, n0, int(1./phi + 0.5), nframe), 'r') as infile:
			e_arrays = np.loadtxt(infile)

		plt.figure(18, figsize=(fig_x,fig_y))
		plt.scatter(Z2, int_exx, c='orange', label=r'$\epsilon_\parallel$', lw=0, s=msize, edgecolors=None)
		plt.scatter(Z2, int_ezz, c='brown', label=r'$\epsilon_\perp$', lw=1, s=msize, marker='x')
		plt.xlabel(r'$z$ (\AA)')
		plt.ylabel(r'$\epsilon(z)$ (a.u)')
		plt.axis([-10., +15, 1, 4.5])
		plt.legend(loc=4)

		for i in xrange(4):
	
			plt.figure(i+19, figsize=(fig_x,fig_y))
			plt.scatter(Z3, e_arrays[i*2], c='orange', label=r'$\epsilon_\parallel$', lw=0, s=msize, edgecolors=None)
			plt.scatter(Z3, e_arrays[i*2+1], c='brown', label=r'$\epsilon_\perp$', lw=1, s=msize, marker='x')
			plt.xlabel(r'$z$ (\AA)')
			plt.ylabel(r'$\epsilon(z)$ (a.u)')
			plt.axis([-10., +15, 1, 2.0])
			plt.legend(loc=4)
			plt.savefig('{}/{}_{}_{}_dielectric.png'.format(groot, model.lower(), cutoff, i))
			plt.savefig('{}/{}_{}_{}_dielectric.pdf'.format(groot, model.lower(), cutoff, i))

	plt.figure(18, figsize=(fig_x,fig_y))
	plt.savefig('{}/{}_{}_int_dielectric.png'.format(groot, model.lower(), cutoff))
	plt.savefig('{}/{}_{}_int_dielectric.pdf'.format(groot, model.lower(), cutoff))
	for i in xrange(4):
		plt.figure(i+19, figsize=(fig_x,fig_y))
		plt.savefig('{}/{}_{}_{}_dielectric.png'.format(groot, model.lower(), cutoff, i))
		plt.savefig('{}/{}_{}_{}_dielectric.pdf'.format(groot, model.lower(), cutoff, i))

	plt.close('all')



def print_graphs_thermodynamics(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, folder, suffix):

	TOT_ENERGY = []
	TOT_ENERGY_ERR = []
	TOT_TEMP = []
	TOT_TEMP_ERR = []
	TOT_Z_RANGE = []
	TOT_A_RANGE = []
	TOT_N_RANGE = []
	TOT_TENSION = []
	TOT_TENSION_ERR = []
	TOT_VAR_TENSION = []
	TOT_AN_RANGE = []
	TOT_DEN = []

	red_units = True
	conv_mJ_kcalmol = 1E-6 / 4.184 * con.N_A

	"Conversion of length and surface tension units"
        if model.upper() == 'ARGON':
                LJ[0] = LJ[0] * 4.184
		if red_units:
			e_constant = 1 / LJ[0]
		        st_constant = ((LJ[1]*1E-10)**2) * con.N_A * 1E-6 / LJ[0]
		        l_constant = 1 / LJ[1]
		else:
			e_constant = 1.
		        st_constant = 1.
		        l_constant = 1E-10
                T = 85
                gam_start = -300 * st_constant
                gam_end = 300 * st_constant
                com = 0
		TEST = ['W', 'A', 'C']
		if folder.upper() == 'SURFACE': 
			nfolder = [60, 0, 7]
			nfolder_plt = [11, 0, 4]
        	elif folder.upper() == 'SURFACE_2': 
			nfolder = [60, 25, 22]
			nfolder_plt = [11, 4, 0]
        	COLOUR = ['b', 'r', 'g']
        	CSIZE = [50, 50, 50]
        else:
                LJ[0] = LJ[0] * 4.184
		e_constant = 1.
                st_constant = 1.
                l_constant = 1E-10
                T = 298
                gam_start = -500 * st_constant
                gam_end = 500 * st_constant
                if model.upper() in ['METHANOL', 'ETHANOL', 'DMSO']: com = 'COM'
                else: com = 0
		TEST = ['W', 'A', 'C']
		nfolder = [40, 0, 0]
		nfolder_plt = [11, 0, 0]
		COLOUR = ['b', 'r', 'g']
		CSIZE = [50, 50, 35]

	for i in xrange(len(TEST)):
		if model.upper() == 'ARGON': root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/{}_TEST'.format(model.upper(), T, cutoff, TEST[i].upper())
        	else: root = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A/{}_TEST'.format(model.upper(), T, cutoff, TEST[i].upper())
		
		(ENERGY, ENERGY_ERR, TEMP, TEMP_ERR, TENSION, TENSION_ERR, VAR_TENSION, N_RANGE, A_RANGE, AN_RANGE, Z_RANGE, 
			DEN) = thermo.energy_tension(root, model, suffix, TEST[i], folder, 
			 nfolder_plt[i], nfolder[i], T, float(cutoff), LJ, CSIZE[i], e_constant, l_constant, st_constant, com, False, False, False)
		
		if model.upper() != 'ARGON' or red_units == False:
			AN_RANGE = AN_RANGE * con.N_A
			#intAN_RANGE = intAN_RANGE * con.N_A
	
		thermo.plot_graphs(ENERGY, ENERGY_ERR, TENSION, TENSION_ERR, VAR_TENSION, N_RANGE, A_RANGE, AN_RANGE,
			Z_RANGE, COLOUR[i], nfolder_plt[i])
		
		for n in xrange(nfolder_plt[i], nfolder[i]):
			TOT_ENERGY.append(ENERGY[n])
			TOT_ENERGY_ERR.append(ENERGY_ERR[n])
			TOT_TEMP.append(TEMP[n])
			TOT_TEMP_ERR.append(TEMP_ERR[n])
			TOT_AN_RANGE.append(AN_RANGE[n])
			TOT_Z_RANGE.append(Z_RANGE[n])
			TOT_A_RANGE.append(A_RANGE[n])
			TOT_N_RANGE.append(N_RANGE[n])
			TOT_TENSION.append(TENSION[n])
			TOT_TENSION_ERR.append(TENSION_ERR[n])
			TOT_VAR_TENSION.append(VAR_TENSION[n])
			TOT_DEN.append(DEN[n])
			
			"""
			if intA_RANGE[n] != 0:
				TOT_int_ENERGY.append(ENERGY[n])
				TOT_ZintA_RANGE.append(ZintA_RANGE[n])
				TOT_intAN.append(intAN_RANGE[n])
				TOT_int_TENSION.append(TENSION[n])
				TOT_int_TENSION_ERR.append(TENSION_ERR[n])
			"""

	c_energy, c_energy_err = thermo.get_U0(model, T, cutoff, 50, LJ[1], LJ[0])
	c_energy *= e_constant
	c_energy_err *= e_constant	

	groot = "/home/fl7g13/Documents/Thesis/Figures/{}_{}_{}".format(model.upper(), csize, cutoff)

	""" FIGURE PARAMETERS """
	fig_x = 12
	fig_y = 8
	msize = 50
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size='20.0')
	plt.rc('lines', linewidth='2.0', markersize=7)
	plt.rc('axes', labelsize='25.0')
	plt.rc('xtick', labelsize='25.0')
	plt.rc('ytick', labelsize='25.0')

	TOT_ZA_RANGE = np.sqrt(np.array([(TOT_Z_RANGE[x] / TOT_A_RANGE[x]) for x in range(len(TOT_A_RANGE))]))
	m, c, r_value, p_value, std_err = stats.linregress( np.sqrt(TOT_ZA_RANGE) , np.array(TOT_TENSION_ERR))
	y_data = [m * x + c for x in np.sqrt(TOT_ZA_RANGE)]

	if model.upper() == 'ARGON' and red_units == True:
		TOT_DEN = np.array(TOT_DEN) / np.sum(M) * con.N_A * 1E-24 / l_constant**3	
	else: TOT_DEN = np.array(TOT_DEN) / np.sum(M) * con.N_A * 1E6

	NI_ARRAY = [thermo.NI_func(TOT_Z_RANGE[x], TOT_A_RANGE[x], TOT_DEN[x], float(cutoff) * l_constant) for x in range(len(TOT_Z_RANGE))]

	print TOT_DEN
	print TOT_VAR_TENSION
	print TOT_A_RANGE
	print NI_ARRAY

	VAR_X = np.array([TOT_VAR_TENSION[x] * 4 * TOT_A_RANGE[x]**2 / NI_ARRAY[x] for x in range(len(TOT_Z_RANGE))]) 

	print VAR_X / e_constant**2

	z_range = np.linspace(np.min(TOT_Z_RANGE), np.max(TOT_Z_RANGE), len(TOT_Z_RANGE))
	a_range = np.linspace(np.min(TOT_A_RANGE), np.max(TOT_A_RANGE), len(TOT_A_RANGE))
	y_data_za = np.sqrt(np.array([thermo.NI_func(TOT_Z_RANGE[x], TOT_A_RANGE[x], np.mean(TOT_DEN), float(cutoff)*l_constant) / 4. * np.mean(VAR_X) / TOT_A_RANGE[x]**2 for x in range(len(TOT_ZA_RANGE))]))
	
	tot_gamma = np.array(TOT_TENSION)
	gamma_err = np.sum([ 1. / TOT_TENSION_ERR[n]**2 for n in xrange(len(TOT_TENSION_ERR))])
	av_gamma = np.sum([ tot_gamma[n] / (TOT_TENSION_ERR[n]**2 * gamma_err) for n in xrange(len(TOT_TENSION_ERR))])

	print "Surface TENSION {} = {} ({})".format(model.upper(), av_gamma, np.sqrt(np.mean(np.array(TOT_TENSION_ERR)**2)))
	if model.upper() == 'ARGON' and red_units == True:
		print "Average var_X = {}".format(np.mean(VAR_X))
		print "Average density = {}".format(np.mean(TOT_DEN))
	else: 
		print "Average var_X = {}".format(np.mean(VAR_X) * conv_mJ_kcalmol**2)
		print "Average density = {}".format(np.mean(TOT_DEN) * np.sum(M) / con.N_A * 1E-6)	
	
	plt.figure(0, figsize=(fig_x,fig_y))
	if model.upper() == 'ARGON':
		if red_units:
			plt.xlabel(r'\LARGE{$L_z^*$ }')
			plt.ylabel(r'\LARGE{$A^*$ }')
			axis = np.array([12, 47, 200, 900])
		else:
			plt.xlabel(r'\LARGE{$L_z$ (\AA)}')
			plt.ylabel(r'\LARGE{$A$ (\AA$^{2}$)}')
			axis = np.array([25, 120, 400, 4000])
	else:	
		plt.xlabel(r'\LARGE{$L_z$ (\AA)}')
		plt.ylabel(r'\LARGE{$A$ (\AA$^{2}$)}')
		if model.upper() == 'TIP4P2005': axis = np.array([25, 120, 400, 4000])
		elif model.upper() == 'SPCE': axis = np.array([25, 120, 400, 4000])
		elif model.upper() == 'TIP3P': axis = np.array([25, 120, 400, 4000])
	plt.axis(axis)
	plt.savefig('{}/{}_sys_dimensions_paper.pdf'.format(groot, model.lower()))

	plt.figure(1, figsize=(fig_x,fig_y))
	if model.upper() == 'ARGON':
		if red_units:
			plt.xlabel(r'\LARGE{$\sqrt{L_z^*/A^*}$ }')
			plt.ylabel(r'\LARGE{$\left<\gamma^*\right>$}')
			axis = np.array([0.10, 0.46, 1.05, 1.23])
		else:
			plt.xlabel(r'\LARGE{$\sqrt{L_z/A}$ (m$^{-\frac{1}{2}}$)}')
			plt.ylabel(r'\LARGE{$\left<\gamma\right>$ (mJ m$^{-2}$)}')
			axis = np.array([5000, 25000, 15.2, 17.6])
	else: 
		plt.xlabel(r'\LARGE{$\sqrt{L_z/A}$ (m$^{-\frac{1}{2}}$)}')
		plt.ylabel(r'\LARGE{$\left<\gamma\right>$ (mJ m$^{-2}$)}')
		if model.upper() == 'TIP4P2005': axis = np.array([14400, 24000, 65, 70.5])
		if model.upper() == 'SPCE': axis = np.array([15500, 25250, 59, 65])
		if model.upper() == 'TIP3P': axis = np.array([14000, 22000, 50, 53])
	plt.axis(axis)
	plt.savefig('{}/{}_surface_tension_paper.pdf'.format(groot, model.lower()))

	plt.figure(2, figsize=(fig_x,fig_y))
	#plt.plot(TOT_ZA_RANGE, y_data, linestyle='dashed', color='black')
	thermo.bubblesort(y_data_za, TOT_ZA_RANGE)
	plt.scatter(TOT_ZA_RANGE, y_data_za, marker='x', color='black', s=100)
	if model.upper() == 'ARGON':
		if red_units:
			plt.xlabel(r'\LARGE{$\sqrt{L_z^*/A^*}$ }')
			plt.ylabel(r'\LARGE{$\sqrt{\mathrm{Var}[\gamma^*]}$}')
			axis = np.array([0.10, 0.46, 0.32, 1.6])
		else:
			plt.xlabel(r'\LARGE{$\sqrt{L_z/A}$ (m$^{-\frac{1}{2}}$)}') 
			plt.ylabel(r'\LARGE{$\sqrt{\mathrm{Var}[\gamma]}$ (mJ m$^{-2}$)}')
			axis = np.array([5000, 25000, 5, 23])
	else:
		plt.xlabel(r'\LARGE{$\sqrt{L_z/A}$ (m$^{-\frac{1}{2}}$)}') 
		plt.ylabel(r'\LARGE{$\sqrt{\mathrm{Var}[\gamma]}$ (mJ m$^{-2}$)}')
		if model.upper() == 'TIP4P2005': axis = np.array([14400, 24000, 80, 135])
		if model.upper() == 'SPCE': axis = np.array([15500, 25250, 82, 140])
		if model.upper() == 'TIP3P': axis = np.array([14000, 22000, 68, 108])
	plt.axis(axis)
	plt.savefig('{}/{}_surface_tension_rms_paper.pdf'.format(groot, model.lower()))

	XAN = np.array(TOT_AN_RANGE) 
	an_range = np.linspace(np.min(XAN), np.max(XAN), len(XAN))
	m, c, r_value, p_value, std_err = stats.linregress( XAN , np.array(TOT_ENERGY))
	ydata = map(lambda x: x * m + c, an_range)

	param, pcov = curve_fit(linear, XAN , np.array(TOT_ENERGY), sigma = TOT_ENERGY_ERR)
	error_Us = np.sqrt(pcov[0][0])
	if model.upper() != 'ARGON' or red_units == False:
		m = param[0] * 1E6
		error_Us = error_Us * 1E6
	print error_Us, std_err, r_value, np.mean(TOT_ENERGY), np.sqrt(np.mean(np.array(TOT_ENERGY_ERR)**2))
	error_Ss = np.sqrt(1. / np.mean(TOT_TEMP)**2 * (error_Us**2 + np.mean(np.array(TOT_TENSION_ERR)**2) + (m - av_gamma)**2 / np.mean(TOT_TEMP)**2 * np.mean(np.array(TOT_TEMP_ERR)**2)))
	ydata = map(lambda x: x * param[0] + param[1], an_range)

	print "\nUsing (U/N) vs (A/N):"
	print "Surface ENERGY {} = {} ({})".format(model.upper(), m , error_Us)
	print "Surface ENTROPY {} = {} ({})".format(model.upper(), (m - av_gamma) / np.mean(TOT_TEMP), error_Ss )
	print "INTERCEPT: {} ({})  CUBIC ENERGY: {} ({})".format(c, np.sqrt(pcov[1][1]), c_energy, c_energy_err)

	plt.figure(5, figsize=(fig_x,fig_y))
	plt.plot(an_range, ydata, linestyle='dashed', color='black')
	#plt.plot(intan_range, iydata, linestyle='dotted', color='black')
	if model.upper() == 'ARGON':
		if red_units:
			plt.xlabel(r'\LARGE{$A^*/N$}')
			plt.ylabel(r'\LARGE{$\left<U^*\right>/N$}')
			axis = np.array([0.04, 0.18, -4.9, -4.5])
		else:
			plt.xlabel(r'\LARGE{$A/N$ (m$^{2}$ mol$^{-1}$)}') 
			plt.ylabel(r'\LARGE{$\left<U\right>/N$ (kJ mol$^{-1}$)}')
			axis = np.array([3000, 12500, -4.85, -4.45])
	else:
		plt.xlabel(r'\LARGE{$A/N$ (m$^{2}$ mol$^{-1}$)}') 
		plt.ylabel(r'\LARGE{$\left<U\right>/N$ (kJ mol$^{-1}$)}')
		if model.upper() == 'TIP4P2005': axis = np.array([3000, 8350, -39.95, -39.3])
		elif model.upper() == 'SPCE': axis = np.array([3250, 8250, -38.9, -38.30])
		elif model.upper() == 'TIP3P': axis = np.array([2900, 7600, -32.25, -31.8])
	plt.axis(axis)
	plt.savefig('{}/{}_surface_energy_paper.pdf'.format(groot, model.lower()))


	plt.figure(6, figsize=(fig_x,fig_y))
	if model.upper() == 'ARGON':
		if red_units:
			plt.xlabel(r'\LARGE{$A^*/N$}')
			plt.ylabel(r'\LARGE{$\left<\gamma^*\right>$}')
			axis = np.array([0.04, 0.18, 1.05, 1.23])
		else:
			plt.xlabel(r'\LARGE{$A/N$ (m$^2$ mol$^{-1}$)}')
			plt.ylabel(r'\LARGE{$\left<\gamma\right>$ (mJ m$^{-2}$)}')
			axis = np.array([3000, 12500, 15.2, 17.6])
	else: 
		plt.xlabel(r'\LARGE{$A/N$ (m$^2$ mol$^{-1}$)}')
		plt.ylabel(r'\LARGE{$\left<\gamma\right>$ (mJ m$^{-2}$)}')
		if model.upper() == 'TIP4P2005': axis = np.array([3000, 8350, 65, 70.5])
		elif model.upper() == 'SPCE': axis = np.array([3250, 8250, 59, 65])
		elif model.upper() == 'TIP3P': axis = np.array([2900, 7600, 50, 53])
	plt.axis(axis)
	plt.savefig('{}/{}_surface_tension2_paper.pdf'.format(groot, model.lower()))
	
	plt.show()


def main(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, TYPE, folder, sfolder, nfolder, suffix):

	print_graphs_thermodynamics(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, folder, suffix)
	#print_graphs_density(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, folder, suffix)
	#print_graphs_orientational(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, folder, suffix)
