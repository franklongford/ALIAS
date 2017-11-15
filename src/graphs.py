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

from scipy import stats
from scipy import constants as con
from scipy.optimize import curve_fit, leastsq
import scipy.integrate as spin
from scipy.interpolate import bisplrep, bisplev, splprep, splev

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
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

lnsp = 25.0


MARKER = ['o', 'v', 'x', '^', 's', 'p', '+', '*']
COLOUR = ['b', 'g', 'r', 'c', 'm', 'saddlebrown', 'navy'] 

def linear(x, m, c): return m*x + c

def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

def print_graphs_density(directory, model, nsite, AT, M, nslice, cutoff, csize, folder, suffix, nframe, DIM):

	groot = "{}/Figures/DEN".format(directory)
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

	plt.close('all')

def print_graphs_intrinsic_density(directory, model, nsite, AT, M, nslice, qm, QM, n0, phi, cutoff, csize, folder, suffix, nframe, DIM, pos_1, pos_2):

	groot = "{}/Figures/INTDEN".format(directory)
	if not os.path.exists(groot): os.mkdir(groot)

	Z1 = np.linspace(0, DIM[2], nslice)
        Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)
	start = int((DIM[2]/2-10) / DIM[2] * nslice)
        stop = int((DIM[2]/2+15) / DIM[2] * nslice)
	skip = (qm-1) / 4

	for r, recon in enumerate([False, True]):

		pos = (np.mean(pos_1[r]) - np.mean(pos_2[r])) / 2.

		Z3 = np.linspace(-DIM[2]/2-pos, DIM[2]/2-pos, nslice)
		Z4 = np.linspace(-DIM[2]-pos, -pos, nslice)

		mol_density_qm = []
		qm_z = []

		with file('{}/DEN/{}_{}_{}_PAR.npy'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
                	param = np.load(infile)
		with file('{}/DEN/{}_{}_{}_DEN.npy'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
                	av_density = np.load(infile)

		Z4 = np.linspace(-DIM[2]/2+param[3], DIM[2]/2+param[3], nslice)

		sm_mol_den = [ut.den_func(z, param[0]/np.sum(M) * con.N_A * 1E-24, param[1]/np.sum(M) * con.N_A * 1E-24, param[2], param[3], param[4]) for z in Z1]
		start = int((DIM[2]/2-param[3]-10) / DIM[2] * nslice)
                stop = int((DIM[2]/2-param[3]+15) / DIM[2] * nslice)

		Zplot = list(Z4[start:stop])
                INTDENplot = list(av_density[-1][start:stop])
		fig = plt.figure(4, figsize=(fig_x+5,fig_y+5))
                ax = fig.gca(projection='3d')
		ax.plot(np.ones(len(Zplot)) * 0, Zplot, INTDENplot, c='g', label=r'$\tilde{\rho}$ (\AA$^{-3}$)')
		EFFDENplot = list(sm_mol_den[start:stop])
		fig = plt.figure(5, figsize=(fig_x+5,fig_y+5))
                ax = fig.gca(projection='3d')
                ax.plot(np.ones(len(Zplot)) * 0, Zplot, EFFDENplot, c='g', label=r'$\hat{\rho}$ (\AA$^{-3}$)')

		if not recon: 
			file_name_den = '{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, n0, int(1/phi + 0.5), nframe)
			figure_name_den = '{}_{}_{}_{}'.format(model.lower(), nslice, qm, nframe)
		else: 
			file_name_den = '{}_{}_{}_{}_{}_{}_R'.format(model.lower(), nslice, qm, n0, int(1/phi + 0.5), nframe)
			figure_name_den = '{}_{}_{}_{}_R'.format(model.lower(), nslice, qm, nframe)

		with file('{}/INTDEN/{}_MOL_DEN.npy'.format(directory, file_name_den), 'r') as infile:
			mol_int_den = np.load(infile)
		with file('{}/INTDEN/{}_EFF_DEN.npy'.format(directory, file_name_den), 'r') as infile:
			mol_eff_den = np.load(infile)
		with file('{}/INTDEN/{}_MOL_DEN_CORR.npy'.format(directory, file_name_den), 'r') as infile:
                        mol_int_den_corr = np.load(infile)

		for j, qm in enumerate(QM):

			mass_int_den = mol_int_den[j] * np.sum(M) / con.N_A * 1E24

			m_size = 40

			if qm % 3 == 0:
				start = int((DIM[2]/2-10) / DIM[2] * nslice)
				stop = int((DIM[2]/2+15) / DIM[2] * nslice)

				Zplot = list(Z2[start:stop])
				INTDENplot = list(mol_int_den[j][start:stop])

				fig = plt.figure(4, figsize=(fig_x+5,fig_y+5))
				ax = fig.gca(projection='3d')
				ax.plot(np.ones(len(Zplot)) * qm, Zplot, INTDENplot, c='g') 

				start = int((DIM[2]/2+pos-10) / DIM[2] * nslice)
				stop = int((DIM[2]/2+pos+15) / DIM[2] * nslice)

				Zplot = list(Z3[start:stop])
				EFFDENplot = list(mol_eff_den[j][start:stop])

				fig = plt.figure(5, figsize=(fig_x+5,fig_y+5))
				ax = fig.gca(projection='3d')
				ax.plot(np.ones(len(Zplot)) * qm, Zplot, EFFDENplot, c='g')

			"""
			Zplot = [-10] + list(Z2[start:stop]) + [15]
			DENplot = [0] + list(mol_int_den[start:stop]) + [0]

			if qm == nm or qm == 1 or qm==nm/2:
				mol_density_qm.append(list(zip(Zplot, DENplot)))
				qm_z.append(qm)
			elif qm == nm/4 or qm == 3*nm/4: 
				mol_density_qm.append(list(zip(Zplot, DENplot)))
				qm_z.append(qm)

			plt.figure(3, figsize=(fig_x,fig_y))
			plt.plot(Z2, mol_int_den)
			plt.xlabel(r'$z$ (\AA)')
			plt.ylabel(r'$\rho(z)$ (\AA$^{-3}$)')
			plt.axis([-10., 15, 0, np.max([np.max(mol_int_den), n0/(DIM[0]*DIM[1])])])
			#plt.legend(loc=3)
			plt.savefig('{}/{}_int_mol_density.png'.format(groot, figure_name_den))
			plt.savefig('{}/{}_int_mol_density.pdf'.format(groot, figure_name_den))
			plt.close(3)

			plt.figure(8, figsize=(fig_x,fig_y))
			plt.plot(Z3, mol_eff_den)
			#plt.plot(np.linspace(0, max_r, rslice), rad_density_array[0], c='green', linestyle='dotted')
			plt.xlabel(r'$z$ (\AA)')
			plt.ylabel(r'$g(z)$')
			plt.axis([-10., 15, 0, 0.08])
			#plt.legend(loc=3)
			plt.savefig('{}/{}_eff_mol_density.png'.format(groot, figure_name_den))
			plt.savefig('{}/{}_eff_mol_density.pdf'.format(groot, figure_name_den))
			plt.close(8)

			plt.figure(4, figsize=(fig_x,fig_y))
			plt.plot(Z2, mass_int_den)
			plt.xlabel(r'$z$ (\AA)')
			plt.ylabel(r'$\rho(z)$ (g cm$^{-3}$)')
			plt.axis([-10., 15, 0, 3.0])
			#plt.legend(loc=3)
			plt.savefig('{}/{}_{}_{}_{}_{}_int_mass_density.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
			plt.savefig('{}/{}_{}_{}_{}_{}_int_mass_density.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))
			

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
		
			"""

		if not recon: figure_name_den = '{}_{}_{}_{}'.format(model.lower(), nslice, qm, nframe)
		else: figure_name_den = '{}_{}_{}_{}_R'.format(model.lower(), nslice, qm, nframe)

		fig = plt.figure(4, figsize=(fig_x+5,fig_y+5))   
		ax = fig.gca(projection='3d')
		#poly = PolyCollection(mol_density_qm, facecolors=[(0.0, 0.5, 0.0, 0.6)] * len(mol_density_qm))
		#poly.set_alpha(0.25)
		#ax.add_collection3d(poly, zs=qm_z, zdir='x')

		ax.set_ylabel(r'$z$ (\AA)', labelpad=lnsp)
		ax.set_ylim3d(-10, 15)
		ax.set_zlabel(r'$\rho(z)$ (\AA$^{-3}$)', labelpad=lnsp)
		ax.set_zlim3d(0, n0/(DIM[0]*DIM[1]))
		ax.set_xlabel(r'$q_m$', labelpad=lnsp)
		ax.set_xlim3d(0, qm)

		ax.xaxis._axinfo['label']['space_factor'] = lnsp
		ax.yaxis._axinfo['label']['space_factor'] = lnsp
		ax.zaxis._axinfo['label']['space_factor'] = lnsp

		fig.tight_layout()

		plt.savefig('{}/{}_int_mol_density.png'.format(groot, figure_name_den))
		plt.savefig('{}/{}_int_mol_density.pdf'.format(groot, figure_name_den))

		fig = plt.figure(5, figsize=(fig_x+5,fig_y+5))
                ax = fig.gca(projection='3d')

                ax.set_ylabel(r'$z$ (\AA)', labelpad=lnsp)
                ax.set_ylim3d(-10, 15)
                ax.set_zlabel(r'$\rho(z)$ (\AA$^{-3}$)', labelpad=lnsp)
                ax.set_zlim3d(0, np.max(mol_eff_den) * 2)
                ax.set_xlabel(r'$q_m$', labelpad=lnsp)
                ax.set_xlim3d(0, qm)
		ax.xaxis._axinfo['label']['space_factor'] = lnsp
		ax.yaxis._axinfo['label']['space_factor'] = lnsp
		ax.zaxis._axinfo['label']['space_factor'] = lnsp

		fig.tight_layout()

                plt.savefig('{}/{}_eff_mol_density.png'.format(groot, figure_name_den))
                plt.savefig('{}/{}_eff_mol_density.pdf'.format(groot, figure_name_den))
                plt.close('all')


def print_graphs_orientational(directory, model, nsite, AT, nslice, a_type, cutoff, csize, folder, suffix, nframe, DIM):

	groot = "{}/Figures/EULER".format(directory)
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
	d_a = 0.06# * av_a

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


def print_graphs_intrinsic_orientational(directory, model, nsite, AT, nslice, qm, QM, n0, phi, a_type, cutoff, csize, folder, suffix, nframe, DIM, pos_1, pos_2):

	groot = "{}/Figures/INTEULER".format(directory)
	if not os.path.exists(groot): os.mkdir(groot)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	start = int((DIM[2]/2-10) / DIM[2] * nslice)
        stop = int((DIM[2]/2+15) / DIM[2] * nslice)
	skip = (qm-1) / 4

	m_size = 40

	eig_val, eig_vec = ut.get_polar_constants(model, a_type)
	av_a = np.mean(eig_val)
	d_a = 0.04

	for r, recon in enumerate([False, True]):

		pos = (np.mean(pos_1[r]) - np.mean(pos_2[r])) / 2.

		Z3 = np.linspace(-DIM[2]/2-pos, DIM[2]/2-pos, nslice)
		Z4 = np.linspace(-DIM[2]-pos, -pos, nslice)

		with file('{}/DEN/{}_{}_{}_PAR.npy'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
                	param = np.load(infile)
        	with file('{}/EULER/{}_{}_{}_{}_EUL.npy'.format(directory, model.lower(), a_type, nslice, nframe), 'r') as infile:
                	axx, azz, av_theta, av_phi, _, P1, P2 = np.load(infile)
		
		Z4 = np.linspace(-DIM[2]/2+param[3], DIM[2]/2+param[3], nslice)

		start = int((DIM[2]/2-param[3]-10) / DIM[2] * nslice)
                stop = int((DIM[2]/2-param[3]+15) / DIM[2] * nslice)
		Zplot = list(Z4[start:stop])
                AXXplot =list(axx[start:stop])
                AZZplot =list(azz[start:stop])

		fig = plt.figure(15, figsize=(fig_x+5,fig_y+5))
		ax = fig.gca(projection='3d')
		ax.scatter(np.ones(len(Zplot)) * 0, Zplot, AXXplot, c='red', label=r'$\alpha_\parallel$', lw=0, s=m_size, alpha = 1)
		ax.scatter(np.ones(len(Zplot)) * 0, Zplot, AZZplot, c='blue', label=r'$\alpha_\perp$', lw=0, s=m_size, alpha = 1)
		plt.legend(loc=4)

		start = int((DIM[2]/2-10) / DIM[2] * nslice)
        	stop = int((DIM[2]/2+15) / DIM[2] * nslice)

		file_name_eul = '{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), a_type, nslice, qm, n0, int(1/phi + 0.5), nframe)
		figure_name_eul = '{}_{}_{}_{}'.format(model.lower(), nslice, qm, nframe)
		if recon: 
			file_name_eul += '_R'
			figure_name_eul += '_R'

		with file('{}/INTEULER/{}_INT_EUL.npy'.format(directory, file_name_eul), 'r') as infile:
			int_axx, int_azz, int_av_theta, int_av_phi, int_P1, int_P2 = np.load(infile)

		for qu in QM:

			"""
			plt.figure(12, figsize=(fig_x,fig_y))
			plt.scatter(Z2, int_av_theta1, c='blue', lw=0, s=m_size)
			plt.scatter(Z2, int_av_phi1, c='red', lw=0, s=m_size)
			plt.xlabel(r'$z$ (\AA)')
			plt.ylabel(r'$\left<\theta\right>(z)$')
			#plt.axis([-6., 8, 1.44, 1.5])
			plt.legend(loc=3)
			plt.savefig('{}/{}_{}_theta1.png'.format(groot, model.lower(), cutoff))

			plt.figure(13, figsize=(fig_x,fig_y))
			plt.scatter(Z2, int_av_theta2, c='blue', lw=0, s=m_size)
			plt.scatter(Z2, int_av_phi2, c='red', lw=0, s=m_size)
			plt.xlabel(r'$z$ (\AA)')
			plt.ylabel(r'$\left<\theta\right>(z)$')
			#plt.axis([-6., 8, 1.44, 1.5])
			plt.savefig('{}/{}_{}_theta2.png'.format(groot, model.lower(), cutoff))
			plt.legend(loc=3)

			plt.figure(14, figsize=(fig_x,fig_y))
			plt.scatter(Z2, int_axx1, c='red', label=r'$\alpha_\parallel$', lw=0, s=m_size)
			plt.scatter(Z2, int_azz1, c='blue', label=r'$\alpha_\perp$', lw=0, s=m_size)
			plt.xlabel(r'$z$ (\AA)')
			plt.ylabel(r'$\alpha(z)$ (\AA$^{3}$)')
			plt.axis([-10, 15, av_a - d_a, av_a + d_a])
			plt.legend(loc=4)
			plt.savefig('{}/{}_{}_{}_{}_{}_int_polarisability_1.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
			plt.savefig('{}/{}_{}_{}_{}_{}_int_polarisability_1.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))

			plt.figure(15, figsize=(fig_x,fig_y))
			plt.scatter(Z2, int_axx2, c='red', label=r'$\alpha_\parallel$', lw=0, s=m_size)
			plt.scatter(Z2, int_azz2, c='blue', label=r'$\alpha_\perp$', lw=0, s=m_size)
			plt.xlabel(r'$z$ (\AA)')
			plt.ylabel(r'$\alpha(z)$ (\AA$^{3}$)')
			plt.axis([-10, 15, av_a - d_a, av_a + d_a])
			plt.legend(loc=3)
			plt.savefig('{}/{}_{}_{}_{}_{}_int_polarisability_2.png'.format(groot, model.lower(), nslice, nm, qm, nframe))
			plt.savefig('{}/{}_{}_{}_{}_{}_int_polarisability_2.pdf'.format(groot, model.lower(), nslice, nm, qm, nframe))
			"""

			Zplot = list(Z2[start:stop])
		        AXXplot =list(int_axx[qu][start:stop])
			AZZplot =list(int_azz[qu][start:stop])

			fig = plt.figure(15, figsize=(fig_x+5,fig_y+5))
			ax = fig.gca(projection='3d')
		        if qu % 3 == 0: 
				ax.scatter(np.ones(len(Zplot)) * qu, Zplot, AXXplot, c='red', lw=0, s=m_size, alpha = 1)
		        	ax.scatter(np.ones(len(Zplot)) * qu, Zplot, AZZplot, c='blue', lw=0, s=m_size, alpha = 1)
				#ax.scatter(Zplot, np.ones(len(Zplot)) * qm, AXXplot, c='red', label=r'$\alpha_\parallel$', lw=0, s=m_size, alpha = 1)
		        	#ax.scatter(Zplot, np.ones(len(Zplot)) * qm, AZZplot, c='blue', label=r'$\alpha_\perp$', lw=0, s=m_size, alpha = 1)
			"""
			elif qm == 1 or qm == nm/2:
				ax.scatter(np.ones(len(Zplot)) * qm, Zplot, AXXplot, c='red', lw=0, s=m_size)
		                ax.scatter(np.ones(len(Zplot)) * qm, Zplot, AZZplot, c='blue', lw=0, s=m_size)
				#ax.scatter(Zplot, np.ones(len(Zplot)) * qm, AXXplot, c='red', lw=0, s=m_size)
		                #ax.scatter(Zplot, np.ones(len(Zplot)) * qm, AZZplot, c='blue', lw=0, s=m_size)
			elif qm == nm/4 or qm == 3*nm/4:
				ax.scatter(np.ones(len(Zplot)) * qm, Zplot, AXXplot, c='red', lw=0, s=m_size)
		                ax.scatter(np.ones(len(Zplot)) * qm, Zplot, AZZplot, c='blue', lw=0, s=m_size)
				#ax.scatter(Zplot, np.ones(len(Zplot)) * qm, AXXplot, c='red', lw=0, s=m_size)
		                #ax.scatter(Zplot, np.ones(len(Zplot)) * qm, AZZplot, c='blue', lw=0, s=m_size)
		        plt.legend(loc=4)

			plt.figure(16, figsize=(fig_x,fig_y))
			plt.scatter(Z2, int_axx, c='red', label=r'$\alpha_\parallel$', lw=0, s=m_size)
			plt.scatter(Z2, int_azz, c='blue', label=r'$\alpha_\perp$', lw=0, s=m_size)
			plt.xlabel(r'$z$ (\AA)')
			plt.ylabel(r'$\alpha(z)$ (\AA$^{3}$)')
			plt.axis([-10, 15, av_a - d_a, av_a + d_a])
			plt.legend(loc=4)
			plt.savefig('{}/{}_int_polarisability.png'.format(groot, figure_name_eul))
			plt.savefig('{}/{}_int_polarisability.pdf'.format(groot, figure_name_eul))
			plt.close(16)
			"""

		if not recon: figure_name_eul = '{}_{}_{}_{}'.format(model.lower(), nslice, qm, nframe)
		else: figure_name_eul = '{}_{}_{}_{}_R'.format(model.lower(), nslice, qm, nframe)

		plt.figure(15, figsize=(fig_x,fig_y))
		ax = fig.gca(projection='3d')
		point  = np.array([0, 0, av_a+d_a])
		normal = np.array([1, 0, 1])
		d = -point.dot(normal)
		X, Y = np.meshgrid(np.linspace(0, 2*d_a, 10), np.linspace(0, qm, 10))
		Z = (-normal[0] * X - normal[1] * Y - d) * 1. / normal[2]
		ax.plot_surface(Y, X, Z, alpha=0.1, color='black')
		ax.plot([0, qm], [0, 0], [av_a, av_a], color='black', linestyle='dashed')
		ax.set_ylabel(r'$z$ (\AA)', labelpad=lnsp)
		ax.set_ylim3d(-10, 15)
		ax.set_zlabel(r'$\alpha(z)$ (\AA$^{3}$)', labelpad=lnsp)
		ax.set_zlim3d(av_a - d_a, av_a + d_a)
		ax.set_xlabel(r'$q_u$', labelpad=lnsp)
		ax.set_xlim3d(0, qm)
		#plt.legend(loc=4)
		plt.savefig('{}/{}_int_polarisability.png'.format(groot, figure_name_eul))
		plt.savefig('{}/{}_int_polarisability.pdf'.format(groot, figure_name_eul))
		plt.close('all')

def print_graphs_dielectric(directory, model, nsite, AT, nslice, a_type, cutoff, csize, folder, suffix, nframe, DIM):

	groot = "{}/Figures/DIELEC".format(directory)
	if not os.path.exists(groot): os.mkdir(groot)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	with file('{}/DATA/DEN/{}_{}_{}_PAR.npy'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
		param = np.load(infile)
	with file('{}/DATA/DIELEC/{}_{}_{}_{}_DIE.npy'.format(directory, model.lower(), a_type, nslice, nframe), 'r') as infile:
		exx, ezz = np.load(infile)

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


def print_graphs_intrinsic_dielectric(directory, model, nsite, AT, nslice, qm, QM, n0, phi, a_type, cutoff, csize, folder, suffix, nframe, DIM, pos_1, pos_2):

	groot = "{}/Figures/INTDIELEC".format(directory)
	if not os.path.exists(groot): os.mkdir(groot)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	with file('{}/DIELEC/{}_{}_{}_{}_DIE_SM.npy'.format(directory, model.lower(), a_type, nslice, nframe), 'r') as infile:
                exx, ezz = np.load(infile)
	
	skip = (qm-1) / 4

	m_size = 40
	eig_val, eig_vec = ut.get_polar_constants(model, a_type)
        av_a = np.mean(eig_val)
	top_e_int = 4.0#(1 + 8 * np.pi / 3. * n0/(DIM[0]*DIM[1]) * av_a) / (1 - 4 * np.pi / 3. * n0/(DIM[0]*DIM[1]) * av_a) 
	top_e_eff = 2.5

	theta = 52.0 / 180 * np.pi

	for r, recon in enumerate([False, True]):

		pos = (np.mean(pos_1[r]) - np.mean(pos_2[r])) / 2.

		Z3 = np.linspace(-DIM[2]/2-pos, DIM[2]/2-pos, nslice)
		Z4 = np.linspace(-DIM[2]-pos, -pos, nslice)

		start = int((DIM[2]/2+pos-10) / DIM[2] * nslice)
        	stop = int((DIM[2]/2+pos+15) / DIM[2] * nslice)
		Zplot = list(Z3[start:stop])
		EXXplot =list(exx[start:stop])
		EZZplot =list(ezz[start:stop])
		NOplot =np.sqrt(exx)[start:stop]
		anis = np.array([1 - (ezz[n] - exx[n]) * np.sin(theta)**2 / ezz[n] for n in range(nslice)])
                NEplot = np.array([np.sqrt(exx[n] / anis[n]) for n in range(nslice)])[start:stop]

		fig = plt.figure(0, figsize=(fig_x+5,fig_y+5))
		ax = fig.gca(projection='3d')
		ax.plot(np.ones(len(Zplot)) * 0, Zplot, EXXplot, c='orange', label=r'$\tilde{\epsilon}_\parallel$')
		ax.plot(np.ones(len(Zplot)) * 0, Zplot, EZZplot, c='brown', label=r'$\tilde{\epsilon}_\perp$', linestyle='dashed')

		fig = plt.figure(1, figsize=(fig_x+5,fig_y+5))
                ax = fig.gca(projection='3d')
                ax.plot(np.ones(len(Zplot)) * 0, Zplot, NOplot, c='purple', label=r'$\tilde{n}_o$')
                ax.plot(np.ones(len(Zplot)) * 0, Zplot, NEplot, c='violet', label=r'$\tilde{n}_e$', linestyle='dashed')

		for i in xrange(3):
			fig = plt.figure(15+i, figsize=(fig_x+5,fig_y+5))
                	ax = fig.gca(projection='3d')
			ax.plot(np.ones(len(Zplot)) * 0, Zplot, EXXplot, c='orange', label=r'$\hat{\epsilon}_\parallel$', alpha = 1)
                	ax.plot(np.ones(len(Zplot)) * 0, Zplot, EZZplot, c='brown', label=r'$\hat{\epsilon}_\perp$',  alpha = 1, linestyle='dashed')


			fig = plt.figure(25+i, figsize=(fig_x+5,fig_y+5))
                        ax = fig.gca(projection='3d')
                        ax.plot(np.ones(len(Zplot)) * 0, Zplot, NOplot, c='purple', label=r'$\hat{n}_o$', alpha = 1)
                        ax.plot(np.ones(len(Zplot)) * 0, Zplot, NEplot, c='violet', label=r'$\hat{n}_e$',  alpha = 1, linestyle='dashed')

		file_name_die = '{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), a_type, nslice, qm, n0, int(1/phi + 0.5), nframe)
		figure_name_die = '{}_{}_{}_{}'.format(model.lower(), nslice, qm, nframe)

		if recon:
			file_name_die += '_R'
			figure_name_die += '_R'

		with file('{}/INTDIELEC/{}_DIE.npy'.format(directory, file_name_die), 'r') as infile:
			int_die = np.load(infile)
		with file('{}/INTDIELEC/{}_CWDIE.npy'.format(directory, file_name_die), 'r') as infile:
			e_arrays = np.load(infile)

		for qu in QM:

			int_exx = int_die[qu][0]
			int_ezz = int_die[qu][1]

			start = int((DIM[2]/2-10) / DIM[2] * nslice)
       			stop = int((DIM[2]/2+15) / DIM[2] * nslice)
			Zplot = list(Z2[start:stop])

			if not np.isnan(np.sqrt(int_exx)).any():
				fig = plt.figure(0, figsize=(fig_x+5,fig_y+5))
				ax = fig.gca(projection='3d')
				EXXplot =list(int_exx[start:stop])
				EZZplot =list(int_ezz[start:stop])
				ax.plot(np.ones(len(Zplot)) * qu, Zplot, EXXplot, c='orange')
				ax.plot(np.ones(len(Zplot)) * qu, Zplot, EZZplot, c='brown', linestyle='dashed')

                                fig = plt.figure(1, figsize=(fig_x+5,fig_y+5))
                                ax = fig.gca(projection='3d')
                                NOplot =np.sqrt(int_exx)[start:stop]
				anis = np.array([1 - (int_ezz[n] - int_exx[n]) * np.sin(theta)**2 / int_ezz[n] for n in range(nslice)])
                		NEplot = np.array([np.sqrt(int_exx[n] / anis[n]) for n in range(nslice)])[start:stop]				

                                ax.plot(np.ones(len(Zplot)) * qu, Zplot, NOplot, c='purple')
                                ax.plot(np.ones(len(Zplot)) * qu, Zplot, NEplot, c='violet', linestyle='dashed')

			start = int((DIM[2]/2+pos-10) / DIM[2] * nslice)
			stop = int((DIM[2]/2+pos+15) / DIM[2] * nslice)
			Zplot = list(Z3[start:stop])
			EXXplot =list(e_arrays[qu][2][start:stop])
			EZZplot =list(e_arrays[qu][3][start:stop])
			NOplot =np.sqrt(e_arrays[qu][2])[start:stop]
                        anis = np.array([1 - (e_arrays[qu][3][n] - e_arrays[qu][2][n]) * np.sin(theta)**2 / e_arrays[qu][3][n] for n in range(nslice)])
                        NEplot = np.array([np.sqrt(e_arrays[qu][2][n] / anis[n]) for n in range(nslice)])[start:stop]

			fig = plt.figure(15, figsize=(fig_x+5,fig_y+5))
                        ax = fig.gca(projection='3d')
			ax.plot(np.ones(len(Zplot)) * qu, Zplot, EXXplot, c='orange')
			ax.plot(np.ones(len(Zplot)) * qu, Zplot, EZZplot, c='brown', linestyle='dashed')

			fig = plt.figure(25, figsize=(fig_x+5,fig_y+5))
                        ax = fig.gca(projection='3d')
                        ax.plot(np.ones(len(Zplot)) * qu, Zplot, NOplot, c='purple')
                        ax.plot(np.ones(len(Zplot)) * qu, Zplot, NEplot, c='violet', linestyle='dashed')

			if not np.isnan(e_arrays[qu][6]).any():
				for i in xrange(2):
					EXXplot =list(e_arrays[qu][4+i*2][start:stop])
                        		EZZplot =list(e_arrays[qu][5+i*2][start:stop])
					NOplot =np.sqrt(e_arrays[qu][4+i*2])[start:stop]
                        		anis = np.array([1 - (e_arrays[qu][5+i*2][n] - e_arrays[qu][4+i*2][n]) * np.sin(theta)**2 / e_arrays[qu][5+i*2][n] for n in range(nslice)])
                        		NEplot = np.array([np.sqrt(e_arrays[qu][4+i*2][n] / anis[n]) for n in range(nslice)])[start:stop]

					fig = plt.figure(16+i, figsize=(fig_x+5,fig_y+5))
                                        ax = fig.gca(projection='3d')
                                	ax.plot(np.ones(len(Zplot)) * qu, Zplot, EXXplot, c='orange')
                                	ax.plot(np.ones(len(Zplot)) * qu, Zplot, EZZplot, c='brown', linestyle='dashed')

					fig = plt.figure(26+i, figsize=(fig_x+5,fig_y+5))
					ax = fig.gca(projection='3d')
					ax.plot(np.ones(len(Zplot)) * qu, Zplot, NOplot, c='purple')
					ax.plot(np.ones(len(Zplot)) * qu, Zplot, NEplot, c='violet', linestyle='dashed')
							

                        """
			plt.figure(18, figsize=(fig_x,fig_y))
			plt.scatter(Z2, int_exx, c='orange', label=r'$\epsilon_\parallel$', lw=0, s=msize, edgecolors=None)
			plt.scatter(Z2, int_ezz, c='brown', label=r'$\epsilon_\perp$', lw=1, s=msize, marker='x')
			plt.xlabel(r'$z$ (\AA)')
			plt.ylabel(r'$\epsilon(z)$ (a.u)')
			plt.axis([-10., +15, 1, 4.5])
			plt.legend(loc=4)
			plt.savefig('{}/{}_int_dielectric.png'.format(groot, figure_name_die))
			plt.savefig('{}/{}_int_dielectric.pdf'.format(groot, figure_name_die))
			plt.close('all')
			

			plt.figure(18, figsize=(fig_x,fig_y))
			plt.plot(Z2, np.sqrt(int_exx), c='purple', label=r'$\tilde{n}_o(z)$')
		        plt.ylabel(r'$n$ (a.u)')
		        plt.axis([-10., +15, 1, 3.0])
		        plt.legend(loc=4)
		        plt.savefig('{}/{}_int_ref_index.png'.format(groot, figure_name_die))
		        plt.savefig('{}/{}_int_ref_index.pdf'.format(groot, figure_name_die))
		        plt.close('all')

			for i in xrange(3):
	
				plt.figure(i+19, figsize=(fig_x,fig_y))
				plt.plot(Z3, e_arrays[(i+1)*2], c='orange', label=r'$\hat{\epsilon}_\parallel$')
				plt.plot(Z3, e_arrays[(i+1)*2+1], c='brown', label=r'$\hat{\epsilon}_\perp$', linestyle='dashed')
				plt.xlabel(r'$z$ (\AA)')
				plt.ylabel(r'$\epsilon$ (a.u)')
				plt.axis([-10., +15, 1, 2.0])
				plt.legend(loc=4)
				plt.savefig('{}/{}_{}_eff_dielectric.png'.format(groot, figure_name_die, i))
				plt.savefig('{}/{}_{}_eff_dielectric.pdf'.format(groot, figure_name_die, i))
				plt.close('all')

				plt.figure(i+19, figsize=(fig_x,fig_y))
		                plt.plot(Z3, np.sqrt(e_arrays[(i+1)*2]), c='purple', label=r'$\hat{n}_o$')
		                plt.xlabel(r'$z$ (\AA)')
		                plt.ylabel(r'$n$ (a.u)')
		                plt.axis([-10., +15, 1, 2.0])
		                plt.legend(loc=4)
		                plt.savefig('{}/{}_{}_eff_ref_index.png'.format(groot, figure_name_die, i))
		                plt.savefig('{}/{}_{}_eff_ref_index.pdf'.format(groot, figure_name_die, i))
		                plt.close('all')
			"""
		if not recon: figure_name_die = '{}_{}_{}_{}'.format(model.lower(), nslice, qm, nframe)
                else: figure_name_die = '{}_{}_{}_{}_R'.format(model.lower(), nslice, qm, nframe)


		fig = plt.figure(0, figsize=(fig_x,fig_y))
		ax = fig.gca(projection='3d')
		ax.set_ylabel(r'$z$ (\AA)', labelpad=lnsp)
		ax.set_ylim3d(-10, 15)
		ax.set_zlabel(r'$\epsilon$', labelpad=lnsp)
		ax.set_zlim3d(1.0, top_e_int)
		ax.set_xlabel(r'$q_u$', labelpad=lnsp)
		ax.set_xlim3d(0, qm)
		plt.legend(loc=4)
		plt.savefig('{}/{}_int_dielectric.png'.format(groot, figure_name_die))
		plt.savefig('{}/{}_int_dielectric.pdf'.format(groot, figure_name_die))

		fig = plt.figure(1, figsize=(fig_x,fig_y))
                ax = fig.gca(projection='3d')
                ax.set_ylabel(r'$z$ (\AA)', labelpad=lnsp)
                ax.set_ylim3d(-10, 15)
                ax.set_zlabel(r'$n$', labelpad=lnsp)
                ax.set_zlim3d(1.0, np.sqrt(top_e_int))
                ax.set_xlabel(r'$q_u$', labelpad=lnsp)
                ax.set_xlim3d(0, qm)
                plt.legend(loc=4)
                plt.savefig('{}/{}_int_ref_index.png'.format(groot, figure_name_die))
                plt.savefig('{}/{}_int_ref_index.pdf'.format(groot, figure_name_die))

		for i in xrange(3):
			fig = plt.figure(15+i, figsize=(fig_x,fig_y))
			ax = fig.gca(projection='3d')
			ax.set_ylabel(r'$z$ (\AA)', labelpad=lnsp)
			ax.set_ylim3d(-10, 15)
			ax.set_zlabel(r'$\epsilon$', labelpad=lnsp)
			ax.set_zlim3d(1.0, top_e_eff)
			ax.set_xlabel(r'$q_u$', labelpad=lnsp)
			ax.set_xlim3d(0, qm)
			plt.legend(loc=4)
			plt.savefig('{}/{}_eff_dielectric_{}.png'.format(groot, figure_name_die, i+1))
			plt.savefig('{}/{}_eff_dielectric_{}.pdf'.format(groot, figure_name_die, i+1))


			fig = plt.figure(25+i, figsize=(fig_x,fig_y))
                        ax = fig.gca(projection='3d')
                        ax.set_ylabel(r'$z$ (\AA)', labelpad=lnsp)
                        ax.set_ylim3d(-10, 15)
                        ax.set_zlabel(r'$n$', labelpad=lnsp)
                        ax.set_zlim3d(1.0, np.sqrt(top_e_eff))
                        ax.set_xlabel(r'$q_u$', labelpad=lnsp)
                        ax.set_xlim3d(0, qm)
                        plt.legend(loc=4)
                        plt.savefig('{}/{}_eff_ref_index_{}.png'.format(groot, figure_name_die, i+1))
                        plt.savefig('{}/{}_eff_ref_index_{}.pdf'.format(groot, figure_name_die, i+1))

		plt.close('all')

def plot_graphs_thermo(ENERGY, ENERGY_ERR, TENSION, TENSION_ERR, VAR_TENSION, N_RANGE, A_RANGE, AN_RANGE, Z_RANGE, l_constant, col):
	"Plots graphs of energy and tension data"

	ZA_RANGE = np.sqrt(Z_RANGE / A_RANGE)
	AZ_RANGE = A_RANGE / Z_RANGE

	plt.figure(0, figsize=(fig_x,fig_y))
	plt.scatter(Z_RANGE / l_constant, A_RANGE / l_constant**2, color=col, s=msize)

	plt.figure(1, figsize=(fig_x,fig_y))
	plt.scatter(ZA_RANGE, TENSION, color=col, marker='x', s=msize)
	plt.errorbar(ZA_RANGE, TENSION, color=col, linestyle='none', yerr=np.array(TENSION_ERR))

	plt.figure(2, figsize=(fig_x,fig_y))
	plt.scatter(np.array(ZA_RANGE), np.sqrt(np.array(VAR_TENSION)), color=col, s=msize)

	plt.figure(3, figsize=(fig_x,fig_y))
	plt.scatter(np.array(AN_RANGE), ENERGY, color=col, s=msize)
	plt.errorbar(np.array(AN_RANGE), ENERGY, color=col, linestyle='none', yerr=np.array(ENERGY_ERR)*5)

	plt.figure(4, figsize=(fig_x,fig_y))
	plt.scatter(np.array(AN_RANGE), TENSION, color=col, s=msize)
	plt.errorbar(np.array(AN_RANGE), TENSION, color=col, linestyle='none', yerr=np.array(TENSION_ERR))


def print_average_graphs_thermo(groot, model, cutoff, csize, red_units, TOT_ZA_RANGE, y_data_za, an_range, ydata):


	plt.figure(0, figsize=(fig_x,fig_y))
	if red_units:
		plt.xlabel(r'\LARGE{$L_l^*$ }')
		plt.ylabel(r'\LARGE{$A^*$ }')
		axis = np.array([12, 47, 200, 900])
		plt.axis(axis)
		plt.savefig('{}/LJ_sys_dimensions_paper.png'.format(groot))
		plt.savefig('{}/LJ_sys_dimensions_paper.pdf'.format(groot))
			
	else:	
		plt.xlabel(r'\LARGE{$L_l$ (\AA)}')
		plt.ylabel(r'\LARGE{$A$ (\AA$^{2}$)}')
		if model.upper() == 'ARGON': axis = np.array([25, 120, 4500, 8000])
		if model.upper() == 'TIP4P2005': axis = np.array([25, 120, 400, 4000])
		elif model.upper() == 'SPCE': axis = np.array([25, 120, 400, 4000])
		elif model.upper() == 'TIP3P': axis = np.array([25, 120, 400, 4000])
		plt.axis(axis)
		plt.savefig('{}/{}_sys_dimensions_paper.png'.format(groot, model.lower()))
		plt.savefig('{}/{}_sys_dimensions_paper.pdf'.format(groot, model.lower()))

	plt.figure(1, figsize=(fig_x,fig_y))
	if red_units:
		plt.xlabel(r'\LARGE{$L_l^*/A^*$ }')
		plt.ylabel(r'\LARGE{$\gamma^*$}')
		axis = np.array([0.10, 0.46, 1.05, 1.23])
		plt.axis(axis)
		plt.savefig('{}/LJ_surface_tension_paper.png'.format(groot))
		plt.savefig('{}/LJ_surface_tension_paper.pdf'.format(groot))

		
	else: 
		plt.xlabel(r'\LARGE{$\sqrt{L_l/A}$ (m$^{-\frac{1}{2}}$)}')
		plt.ylabel(r'\LARGE{$\gamma$ (mJ m$^{-2}$)}')
		if model.upper() == 'ARGON': axis = np.array([4000, 16000, 15.0, 17.0])
		if model.upper() == 'TIP4P2005': axis = np.array([10000, 17000, 65, 70.5])
		if model.upper() == 'SPCE': axis = np.array([11000, 18000, 59, 65])
		if model.upper() == 'TIP3P': axis = np.array([14000, 22000, 50, 53])
		plt.axis(axis)
		plt.savefig('{}/{}_surface_tension_paper.png'.format(groot, model.lower()))
		plt.savefig('{}/{}_surface_tension_paper.pdf'.format(groot, model.lower()))

	plt.figure(2, figsize=(fig_x,fig_y))
	#plt.plot(TOT_ZA_RANGE, y_data, linestyle='dashed', color='black')
	ut.bubblesort(y_data_za, TOT_ZA_RANGE)
	plt.scatter(TOT_ZA_RANGE, y_data_za, marker='x', color='black', s=100)

	if red_units:
		plt.xlabel(r'\LARGE{$\sqrt{L_l^*/A^*}$ }')
		plt.ylabel(r'\LARGE{$\sqrt{\mathrm{Var}[\gamma^*]}$}')
		axis = np.array([0.10, 0.46, 0.32, 1.6])
		plt.axis(axis)
		plt.savefig('{}/LJ_surface_tension_rms_paper.png'.format(groot))
		plt.savefig('{}/LJ_surface_tension_rms_paper.pdf'.format(groot))
			
	else:
		plt.xlabel(r'\LARGE{$\sqrt{L_l/A}$ (m$^{-\frac{1}{2}}$)}') 
		plt.ylabel(r'\LARGE{$\sqrt{\mathrm{Var}[\gamma]}$ (mJ m$^{-2}$)}')
		if model.upper() == 'ARGON': axis = np.array([4000, 16000, 5, 20])
		elif model.upper() == 'TIP4P2005': axis = np.array([10000, 17000, 80, 135])
		elif model.upper() == 'SPCE': axis = np.array([11000, 18000, 82, 140])
		elif model.upper() == 'TIP3P': axis = np.array([14000, 22000, 68, 108])
		plt.axis(axis)
		plt.savefig('{}/{}_surface_tension_rms_paper.png'.format(groot, model.lower()))
		plt.savefig('{}/{}_surface_tension_rms_paper.pdf'.format(groot, model.lower()))

	plt.figure(3, figsize=(fig_x,fig_y))
	plt.plot(an_range, ydata, linestyle='dashed', color='black')
	#plt.plot(intan_range, iydata, linestyle='dotted', color='black')
	if red_units:
		plt.xlabel(r'\LARGE{$A^*/N$}')
		plt.ylabel(r'\LARGE{$\left<U^*\right>/N$}')
		axis = np.array([0.04, 0.18, -4.65, -4.25])
		plt.axis(axis)
		plt.savefig('{}/LJ_surface_energy_paper.png'.format(groot))
		plt.savefig('{}/LJ_surface_energy_paper.pdf'.format(groot))	
			
	else:
		plt.xlabel(r'\LARGE{$A/N$ (m$^{2}$ mol$^{-1}$)}') 
		plt.ylabel(r'\LARGE{$\left<U\right>/N$ (kJ mol$^{-1}$)}')
		if model.upper() == 'ARGON': axis = np.array([4500, 12000, -4.65, -4.30])
		if model.upper() == 'TIP4P2005': axis = np.array([3250, 8350, -39.85, -39.25])
		elif model.upper() == 'SPCE': axis = np.array([3250, 8250, -38.85, -38.30])
		elif model.upper() == 'TIP3P': axis = np.array([2900, 7600, -32.25, -31.8])
		plt.axis(axis)
		plt.savefig('{}/{}_surface_energy_paper.png'.format(groot, model.lower()))
		plt.savefig('{}/{}_surface_energy_paper.pdf'.format(groot, model.lower()))


	plt.figure(4, figsize=(fig_x,fig_y))
	if red_units:
		plt.xlabel(r'\LARGE{$A^*/N$}')
		plt.ylabel(r'\LARGE{$\gamma^*$}')
		axis = np.array([0.04, 0.18, 1.05, 1.23])
		plt.axis(axis)
		plt.savefig('{}/LJ_surface_tension2_paper.png'.format(groot))
		plt.savefig('{}/LJ_surface_tension2_paper.pdf'.format(groot))
			
	else: 
		plt.xlabel(r'\LARGE{$A/N$ (m$^2$ mol$^{-1}$)}')
		plt.ylabel(r'\LARGE{$\gamma$ (mJ m$^{-2}$)}')
		if model.upper() == 'ARGON': axis = np.array([4500, 12500, 15.0, 17.0])
		if model.upper() == 'TIP4P2005': axis = np.array([3350, 8350, 65, 70.5])
		elif model.upper() == 'SPCE': axis = np.array([3250, 8250, 59, 65])
		elif model.upper() == 'TIP3P': axis = np.array([2900, 7600, 50, 53])
		plt.axis(axis)
		plt.savefig('{}/{}_surface_tension2_paper.png'.format(groot, model.lower()))
		plt.savefig('{}/{}_surface_tension2_paper.pdf'.format(groot, model.lower()))



def main(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, TYPE, folder, sfolder, nfolder, suffix):

	print_graphs_thermodynamics(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, folder, suffix)
	#print_graphs_density(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, folder, suffix)
	#print_graphs_orientational(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, folder, suffix)
