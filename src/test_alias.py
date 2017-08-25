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
import thermodynamics as thermo
import intrinsic_sampling_method as ism
import mdtraj as md


model = 'methanol'

if model.upper() == 'ARGON':
	T = 85
	cutoff = 10
	root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/SLAB'.format(model.upper(), T, cutoff)
	com = '0'
	a_type = 'exp'
	folder = 'SURFACE_2'


elif model.upper() == 'METHANOL':
	T = 298
	cutoff = 22
	root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/SLAB'.format(model.upper(), T, cutoff)
	com = 'COM'
	a_type = 'calc'
	folder = 'SURFACE'
	
	
else:
	T = 298
	cutoff = 10
	root = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A/SLAB'.format(model.upper(), T, cutoff)
	com = '0'
	a_type = 'exp'
	folder = 'SURFACE_2'


suffix = 'surface'

a = ut.get_polar_constants(model, a_type)

csize = 80
directory = '{}/{}'.format(root, folder.upper())

nsite, AT, Q, M, LJ = ut.get_model_param(model)
natom, nmol, ntraj, DIM, COM = ut.get_sim_param(root, directory, model, nsite, suffix, csize, M, com, False)
e_constant, st_constant, l_constant = ut.get_thermo_constants(model, LJ)

assert ntraj == 4000

epsilon = np.max(LJ[0]) * 4.184
sigma = np.max(LJ[1])

lslice = 0.05 * sigma
nslice = int(DIM[2] / lslice)
vlim = 3
ncube = 3

mol_sigma, ns, phi = ut.get_ism_constants(model, sigma)

n0 = int(DIM[0] * DIM[1] * ns / mol_sigma**2)

qu = 2 * np.pi / mol_sigma
ql = 2 * np.pi / np.sqrt(DIM[0] * DIM[1])
nm = int(qu / ql)

A = DIM[0] * DIM[1] * l_constant**2 


def plot_ism_surface():

	nframe = 1
	qm = nm
	nxy = 40 
	phi = 5E-8

	X = np.linspace(0, DIM[0], nxy)
	Y = np.linspace(0, DIM[1], nxy)
	
	for frame in xrange(nframe):

		xat, yat, zat = ut.read_atom_positions(directory, model, csize, frame)
		xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)

		auv1, auv2, piv_n1, piv_n2 = ism.intrinsic_surface(directory, model, csize, nsite, nmol, ncube, DIM, COM, nm, n0, phi, vlim, mol_sigma, M, frame, nframe, False)	

		piv_x1 = [xmol[piv] for piv in piv_n1]
		piv_y1 = [ymol[piv] for piv in piv_n1]
		piv_z1 = [zmol[piv] - COM[frame][2] for piv in piv_n1] 

		piv_x2 = [xmol[piv] for piv in piv_n2]
		piv_y2 = [ymol[piv] for piv in piv_n2]
		piv_z2 = [zmol[piv] - COM[frame][2] for piv in piv_n2] 

		surface1 = np.zeros((nxy, nxy))
		surface2 = np.zeros((nxy, nxy))
		plane = np.ones((nxy, nxy)) 

		for i, x in enumerate(X):
			for j, y in enumerate(Y):
				#plane[i][j] += (xi - x) * dzx + (yi - y) * dzy
				surface1[i][j] += ism.xi(x, y, nm, qm, auv1, DIM)
				surface2[i][j] += ism.xi(x, y, nm, qm, auv2, DIM) 

		fig = plt.figure(0, figsize=(15,15))
		ax = fig.gca(projection='3d')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		X, Y = np.meshgrid(X, Y)
		ax.plot_wireframe(Y, X, surface1, color='r')
		ax.scatter(piv_x1, piv_y1, piv_z1)

		fig = plt.figure(1, figsize=(15,15))
		ax = fig.gca(projection='3d')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		ax.plot_wireframe(Y, X, surface2, color='r')
		ax.scatter(piv_x2, piv_y2, piv_z2)
		plt.show()


def test_ism_density():

	nframe = 1
	QM = [0]
	
	for frame in xrange(nframe):

		auv1, auv2, piv_n1, piv_n2 = ism.intrinsic_surface(directory, model, csize, nsite, nmol, ncube, DIM, COM, nm, n0, phi, vlim, mol_sigma, M, frame, nframe, False)	

		with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTZ_AT.txt'.format(directory, model.lower(), nm, qm, n0, int(1/phi + 0.5), frame), 'r') as infile:
                	old_int_z_at_1, old_int_z_at_2 = np.loadtxt(infile)

		int_z_at_1, int_z_at_2 = ism.intrinsic_positions(directory, model, csize, frame, auv1, auv2, natom, nmol, nsite, nm, QM, n0, phi, DIM)

		assert np.sum(old_int_z_at_1 - int_z_at_1) == 0
		assert np.sum(old_int_z_at_2 - int_z_at_2) == 0
		
		#intrinsic_dxdyz(directory, model, csize, frame, auv1, auv2, nmol, nsite, nm, QM, n0, phi, DIM)
"""
def test_orientation():

	xat, yat, zat = ut.read_atom_positions(directory, model, csize, 0)
	xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, 0)
	xR, yR, zR = COM[0]

	with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTDXDY_MOL.txt'.format(directory, model.lower(), nm, nm, n0, int(1./phi + 0.5), 0), 'r') as infile:
		dxdyz_mol = np.loadtxt(infile)

	colour = ['red', 'green', 'cyan']

	for j in xrange(nmol):
		molecule = np.zeros((nsite, 3))

		dzx1 = dxdyz_mol[0][j]
		dzy1 = dxdyz_mol[1][j]
		dzx2 = dxdyz_mol[2][j]
		dzy2 = dxdyz_mol[3][j]

		print dzx1, dzy1

		for l in xrange(nsite):
			molecule[l][0] = xat[j*nsite+l]
			molecule[l][1] = yat[j*nsite+l]
			molecule[l][2] = zat[j*nsite+l] - zR

		O = ut.local_frame_molecule(molecule, model)
		if O[2][2] < -1: O[2][2] = -1.0
		elif O[2][2] > 1: O[2][2] = 1.0

		T = ut.local_frame_surface(dzx1, dzy1, 1)
		R1 = np.dot(O, np.linalg.inv(T))
		O_ = np.dot(T, R1)
		if R1[2][2] < -1: R1[2][2] = -1.0
		elif R1[2][2] > 1: R1[2][2] = 1.0

		fig = plt.figure(0, figsize=(15,15))
		ax = fig.gca(projection='3d')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		for l in xrange(nsite): ax.scatter(molecule[l][0], molecule[l][1], molecule[l][2], color='red')
		ax.plot([molecule[0][0], molecule[1][0]], [molecule[0][1], molecule[1][1]], [molecule[0][2], molecule[1][2]], color='red')
		ax.plot([molecule[0][0], molecule[2][0]], [molecule[0][1], molecule[2][1]], [molecule[0][2], molecule[2][2]], color='red')

		for l in xrange(3): 
			ax.plot([xmol[j], xmol[j] + O_[0][l]], [ymol[j], ymol[j] + O_[1][l]], [zmol[j] - zR, zmol[j] - zR + O_[2][l]] , color='black')
			ax.plot([xmol[j], xmol[j] + T[0][l]], [ymol[j], ymol[j] + T[1][l]], [zmol[j] - zR, zmol[j] - zR + T[2][l]] , color=colour[l], linestyle='dashed')

		X = np.linspace(xmol[j]-1.5, xmol[j]+1.5, 10)
		Y = np.linspace(ymol[j]-1.5, ymol[j]+1.5, 10)

		plane = np.ones((len(X), len(Y))) * (zmol[j] - zR)

		for n, xi in enumerate(X):
			for m, yi in enumerate(Y):
				plane[n][m] += (xi - xmol[j]) * dzx1 + (yi - ymol[j]) * dzy1
				ax.scatter(xi, yi, plane[n][m], color='b')

		ax.set_xlim(xmol[j]-1.5, xmol[j]+1.5)
		ax.set_ylim(ymol[j]-1.5, ymol[j]+1.5)
		ax.set_zlim(zmol[j]-zR-1.5, zmol[j]-zR+1.5)
		#X, Y = np.meshgrid(X, Y)
		#ax.plot_wireframe(Y, X, plane, color='b')
		
		plt.show()

def test_thermo():
	
	with file('{}/DATA/DEN/{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, ntraj), 'r') as infile:
		av_density = np.loadtxt(infile)

	corr_e = ut.E_janecek(av_density[-1], float(cutoff), sigma, epsilon, lslice, A /  l_constant**2) 
	corr_st = ut.ST_janecek(av_density[-1], float(cutoff), sigma, epsilon, lslice) * 1E26 / con.N_A
	
	assert np.round(corr_e, 5) == -190.87578
	assert np.round(corr_st, 5) == 5.37422
	
	FILE = '{}/{}_{}_{}'.format(directory, model.lower(), csize, suffix)
	E, POT, T_, T_err, ST, TOTAL_ENERGY, TOTAL_POTENTIAL, TOTAL_TENSION, TOTAL_TEMP = ut.read_energy_temp_tension(FILE)

	assert E == -23045.9539
	assert POT == -27523.0412
	assert T_ == 298.01
	assert T_err == 3.44
	assert ST == 56.6222
	assert len(TOTAL_ENERGY) == 400000
	assert len(TOTAL_POTENTIAL) == 400000
	assert len(TOTAL_TENSION) == 400000
	assert len(TOTAL_TEMP) == 400000

	ntb = int(len(TOTAL_TENSION) / 100)

	ENERGY_ERR, POTENTIAL_ERR, TENSION_ERR = ut.get_block_error_thermo(TOTAL_ENERGY, TOTAL_POTENTIAL, TOTAL_TENSION, directory, model, csize, ntraj, ntb, False)

	assert np.round(ENERGY_ERR, 5) == 1.35597
	assert np.round(POTENTIAL_ERR, 5) ==  1.28622
	assert np.round(TENSION_ERR, 5) == 0.45261
"""

plot_ism_surface()
