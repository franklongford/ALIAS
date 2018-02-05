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


def print_graphs_intrinsic_density(directory, graph_dir, file_name, nsite, AT, M, nslice, qm, QM, n0, phi, nframe, DIM, pos_1, pos_2):

	groot = "{}/INTDEN".format(graph_dir)
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

		if not recon: 
			file_name_den = '{}_{}_{}_{}_{}_{}'.format(file_name, nslice, qm, n0, int(1/phi + 0.5), nframe)
			figure_name_den = '{}_{}_{}_{}'.format(file_name, nslice, qm, nframe)
		else: 
			file_name_den = '{}_{}_{}_{}_{}_{}_R'.format(file_name, nslice, qm, n0, int(1/phi + 0.5), nframe)
			figure_name_den = '{}_{}_{}_{}_R'.format(file_name, nslice, qm, nframe)

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


		if not recon: figure_name_den = '{}_{}_{}_{}'.format(file_name, nslice, qm, nframe)
		else: figure_name_den = '{}_{}_{}_{}_R'.format(file_name, nslice, qm, nframe)

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



