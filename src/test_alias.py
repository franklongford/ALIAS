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

""" FIGURE PARAMETERS """
fig_x = 12
fig_y = 8
msize = 50
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='20.0')
plt.rc('lines', linewidth='1.0', markersize=7)
plt.rc('axes', labelsize='25.0')
plt.rc('xtick', labelsize='25.0')
plt.rc('ytick', labelsize='25.0')

model = 'argon'

if model.upper() == 'ARGON':
	T = 85
	cutoff = 10
	root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/T_TEST'.format(model.upper(), T, cutoff)
	#root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/SLAB'.format(model.upper(), T, cutoff)
	com = '0'
	a_type = 'exp'
	folder = 'SURFACE_2'
	#csize = 80
	csize = 50
	phi = 5E-6

elif model.upper() == 'METHANOL':
	T = 298
	cutoff = 22
	root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/SLAB'.format(model.upper(), T, cutoff)
	com = 'COM'
	a_type = 'calc'
	folder = 'SURFACE'
	csize = 100
	phi = 5E-8
	
else:
	T = 298
	cutoff = 10
	root = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A/SLAB'.format(model.upper(), T, cutoff)
	com = '0'
	a_type = 'exp'
	folder = 'SURFACE_2'
	csize = 80
	phi = 5E-8

suffix = 'surface'

a = ut.get_polar_constants(model, a_type)

directory = '{}/{}'.format(root, folder.upper())

nsite, AT, Q, M, LJ = ut.get_model_param(model)
natom, nmol, ntraj, DIM, COM = ut.get_sim_param(root, directory, model, nsite, suffix, csize, M, com, False, True)
e_constant, st_constant, l_constant, p_constant, T_constant = ut.get_thermo_constants(model, LJ)

epsilon = np.max(LJ[0]) * 4.184
sigma = np.max(LJ[1])

lslice = 0.05 * sigma
nslice = int(DIM[2] / lslice)
vlim = 3
ncube = 10

mol_sigma, ns = ut.get_ism_constants(model, sigma)

n0 = int(DIM[0] * DIM[1] * ns / mol_sigma**2)

qu = 2 * np.pi / mol_sigma
ql = 2 * np.pi / np.sqrt(DIM[0] * DIM[1])
nm = int(qu / ql)

A = 2 * DIM[0] * DIM[1] * l_constant**2
Vslice = DIM[1] * DIM[2] * lslice


def plot_ism_surface():

	nframe = 100
	qm = nm/2
	nxy = 40 

	X = np.linspace(0, DIM[0], nxy)
	Y = np.linspace(0, DIM[1], nxy)
	
	av_den_corr_matrix = np.zeros((nslice, 100))
        av_z_nz_matrix = np.zeros((100, 100))

	ow_pos = False
	ow_dxdyz = False
	ow_count = False

	av_auv1_2 = np.zeros((2*nm+1)**2)
	av_auv2_2 = np.zeros((2*nm+1)**2)

	av_auv1 = np.zeros(nframe)
	av_auv2 = np.zeros(nframe)

	qm_xy = np.arange(-qm, qm+1)

	for frame in xrange(nframe):
		sys.stdout.write("PROCESSING FRAME {}\r".format(frame))
		sys.stdout.flush()

		xat, yat, zat = ut.read_atom_positions(directory, model, csize, frame)
		xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)

		auv1, auv2, piv_n1, piv_n2 = ism.intrinsic_surface(directory, model, csize, nsite, nmol, ncube, DIM, COM, nm, n0, phi, vlim, mol_sigma, M, frame, nframe, False)

		av_auv1_2 += auv1**2 / nframe
		av_auv2_2 += auv2**2 / nframe

		av_auv1[frame] = auv1[len(auv1)/2]
		av_auv2[frame] = auv2[len(auv2)/2]

		if ow_pos: intrinsic_positions(directory, model, csize, frame, auv1, auv2, nsite, nm, nm, n0, phi, DIM, ow_pos)
                if ow_dxdyz: intrinsic_dxdyz(directory, model, csize, frame, auv1, auv2, nsite, nm, nm, n0, phi, DIM, ow_dxdyz) 

		int_count_corr_array, int_count_z_nz = ism.intrinsic_z_den_corr(directory, COM, model, csize, nm, qm, n0, phi, frame, nslice, nsite, DIM, ow_count)
		av_den_corr_matrix += int_count_corr_array
		av_z_nz_matrix += int_count_z_nz

		#"""
		if frame == nframe-1: 
			auto_corr = ism.auv_xy_correlation(auv1**2, nm, qm)

			X_grid, Y_grid = np.meshgrid(np.linspace(-DIM[0]/2, DIM[0]/2, 2*qm+1), np.linspace(-DIM[1]/2, DIM[1]/2, 2*qm+1))
			fig = plt.figure(0, figsize=(10,5))
			ax = fig.gca(projection='3d')
			ax.plot_wireframe(X_grid, Y_grid, auto_corr)
		
			#piv_x1 = [xmol[piv] for piv in piv_n1]
			#piv_y1 = [ymol[piv] for piv in piv_n1]
			#piv_z1 = [zmol[piv] - COM[frame][2] for piv in piv_n1] 

			piv_x2 = [xmol[piv] for piv in piv_n2]
			piv_y2 = [ymol[piv] for piv in piv_n2]
			piv_z2 = [zmol[piv] - COM[frame][2] for piv in piv_n2] 

			#surface1 = np.zeros((nxy, nxy))
			surface2 = np.zeros((nxy, nxy))
			#plane = np.ones((nxy, nxy)) 

			for i, x in enumerate(X):
				for j, y in enumerate(Y):
					#plane[i][j] += (xi - x) * dzx + (yi - y) * dzy
					#surface1[i][j] += ism.xi(x, y, nm, qm, auv1, DIM)
					surface2[i][j] += ism.xi(x, y, nm, qm, auv2, DIM) 

			fig = plt.figure(1, figsize=(15,15))
			ax = fig.gca(projection='3d')
			ax.set_xlabel(r'$x$ (\AA)')
			ax.set_ylabel(r'$y$ (\AA)')
			ax.set_zlabel(r'$z$ (\AA)')
			ax.set_xlim3d(0, DIM[0])
			ax.set_ylim3d(0, DIM[1])
			X_grid, Y_grid = np.meshgrid(X, Y, alpha=0.75)
			ax.plot_wireframe(Y_grid, X_grid, surface2, color='r')
			ax.scatter(piv_x2, piv_y2, piv_z2)

			#fig = plt.figure(2, figsize=(15,15))
			#ax = fig.gca(projection='3d')
			#ax.set_xlabel('X')
			#ax.set_ylabel('Y')
			#ax.set_zlabel('Z')

			#ax.plot_wireframe(Y, X, surface2, color='r')
			#ax.scatter(piv_x2, piv_y2, piv_z2)


	N = np.linspace(0, 50 * lslice, 100)
	NZ = np.linspace(0, 1, 100)
	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	"""
	Axi = ism.slice_area(av_auv_2, nm, qm, DIM)
	An_corr = np.array([ism.area_correction(z, av_auv_2, nm, qm, DIM) for z in Z2])

	print Delta, Axi, np.max(An_corr), Z2[np.argmax(An_corr)], ism.area_correction(0, av_auv_2, nm, qm, DIM)
	plt.plot(Z2, An_corr)
	plt.show()
	"""

	P_z_nz = av_z_nz_matrix * 2 / (np.sum(av_z_nz_matrix) * 0.01 * lslice)
	P_den_corr_matrix = av_den_corr_matrix / np.sum(av_den_corr_matrix)
	P_corr = np.array([np.sum(A) for A in np.transpose(P_den_corr_matrix)])
	int_den_corr = av_den_corr_matrix / (2 * nframe * Vslice)

	sample = 5
	full_den = np.zeros(nslice)
	norm_int_den = np.zeros((100, nslice))
	plot_norm_int_den = np.zeros((nslice/sample, 100))

	for n3, z3 in enumerate(NZ):
		if P_corr[n3] > 0.075 * np.max(P_corr):
			temp_den = np.transpose(int_den_corr)[n3] / P_corr[n3]
			full_den += temp_den
			norm_int_den[n3] += temp_den

	full_den *= 1. / len(np.argwhere(P_corr > 0.075 * np.max(P_corr)))
	av_den = np.array([np.sum(A)for A in int_den_corr])
	norm_int_den = np.transpose(norm_int_den)

	for n2, z2 in enumerate(Z2):
		try: plot_norm_int_den[n2/sample] += norm_int_den[n2] #/ An_corr[n2]
		except: pass

	min_index = np.max(np.argwhere(P_corr<= 0.075 * np.max(P_corr)))

	plot_norm_int_den = np.transpose(np.transpose(plot_norm_int_den)[min_index:])
	plot_norm_int_den = plot_norm_int_den / np.max(plot_norm_int_den)

	X_grid, Y_grid = np.meshgrid(NZ[min_index:], np.linspace(-DIM[2]/2, DIM[2]/2, nslice/(sample)))
	fig = plt.figure(2, figsize=(10,5))
	ax = fig.gca(projection='3d')
	ax.plot_surface(X_grid, Y_grid, plot_norm_int_den, color='b', linewidth=0.01, antialiased=False, cmap=cm.coolwarm, cstride=1, rstride=2)
	#ax.plot(np.zeros(nslice/sample), np.zeros(100-min_index), [(x)**4 for x in NZ[min_index:]], color='g', alpha = 0.8, linewidth=1)
	ax.set_xlim(NZ[min_index], 1)
	ax.set_ylim(-DIM[2]/2, DIM[2]/2)
	ax.set_zlim(0, 1)

	fig = plt.figure(3, figsize=(10,5))
	plt.plot(Z2, full_den)
	plt.plot(Z2, av_den)

	cw_array = np.zeros((2, nslice))

	Delta1 = (ut.sum_auv_2(av_auv1_2, nm, qm) - np.mean(av_auv1)**2)
	Delta2 = (ut.sum_auv_2(av_auv1_2, nm, qm) - np.mean(av_auv1)**2)
			
	STD = np.sqrt(0.5 * (Delta1 + Delta2))
	length = int(STD / lslice) * 12
	ZG = np.arange(-lslice*length/2, lslice*length/2 + lslice/2, lslice)
	P_den = [ut.gaussian(z, 0, STD) for z in ZG]

	for n1, z1 in enumerate(Z1):
		for n2, z2 in enumerate(ZG):
			sys.stdout.write("PERFORMING GAUSSIAN SMOOTHING {0:.1%} COMPLETE \r".format(float(n1 * nslice + n2) / nslice**2) )
			sys.stdout.flush()

			index = int((z1 - z2) / DIM[2] * nslice) % nslice
			try:
				cw_array[0][n1] += full_den[index] * P_den[n2] * lslice
				cw_array[1][n1] += av_den[index] * P_den[n2] * lslice
			except IndexError: pass

	fig = plt.figure(4, figsize=(10,5))
	plt.plot(Z2, cw_array[0])
	plt.plot(Z2, cw_array[1])

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
"""
sample = 5
full_den = np.zeros(nslice)
norm_int_den = np.zeros((100, nslice))
plot_norm_int_den = np.zeros((nslice/sample, 100))
for n3, z3 in enumerate(NZ):
	if P_corr[n3] > 0.075 * np.max(P_corr):
		temp_den = np.transpose(int_den_corr)[n3] / P_corr[n3]
		full_den += temp_den
		norm_int_den[n3] += temp_den

norm_int_den = np.transpose(norm_int_den)

for n2, z2 in enumerate(Z2):
	try: plot_norm_int_den[n2/sample] += norm_int_den[n2]
	except: pass

min_index = np.max(np.argwhere(P_corr<= 0.075 * np.max(P_corr)))

plot_norm_int_den = np.transpose(np.transpose(plot_norm_int_den)[min_index:])

X_grid, Y_grid = np.meshgrid(NZ, N)

print plot_norm_int_den.shape, np.max(plot_norm_int_den)

plot_norm_int_den = plot_norm_int_den / np.max(plot_norm_int_den)


fig = plt.figure(1, figsize=(10,5))
ax = fig.gca(projection='3d')
ax.plot_wireframe(X_grid, Y_grid, P_z_nz, alpha=0.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 50 * lslice)
ax.set_zlim(0, 0.1)

if qm == nm:
	X_grid, Y_grid = np.meshgrid(NZ[min_index:], np.linspace(-DIM[2]/2, DIM[2]/2, nslice/(sample)))
	fig = plt.figure(2, figsize=(10,5))
	ax = fig.gca(projection='3d')
	ax.plot_surface(X_grid, Y_grid, plot_norm_int_den, color='b', linewidth=0.01, antialiased=False, cmap=cm.coolwarm, cstride=1, rstride=2)
	#ax.plot(np.zeros(nslice/sample), np.zeros(100-min_index), [(x)**4 for x in NZ[min_index:]], color='g', alpha = 0.8, linewidth=1)
	ax.set_xlim(NZ[min_index], 1)
	ax.set_ylim(-DIM[2]/2, DIM[2]/2)
	ax.set_zlim(0, 1)

	plt.show()
#plt.savefig('/home/fl7g13/Documents/Thesis/Figures/{}_{}_10/{}_{}_{}_{}_{}_{}_P_z_nz.png'.format(model.upper(), csize, model.lower(), nm, qm, n0, int(1/phi + 0.5), nframe))

print np.max(P_z_nz), np.sum(P_z_nz) * 0.01 * lslice / 2


P_z = np.array([np.sum(A) * 0.01 for A in P_z_nz])
b_est = np.mean(np.abs(P_corr - np.median(P_corr)))

min_index = np.max(np.argwhere(P_corr<=P_corr[-1]/2))

print min_index, P_corr[-1]/2, P_corr[min_index], P_corr[min_index+1]

for n3, z3 in enumerate(NZ):
	if P_corr[n3] > 0.1 * np.max(P_corr): 
		temp_den = np.transpose(int_den_corr)[n3] / P_corr[n3]
		full_den += temp_den

full_den *= 1. / len(np.argwhere(P_corr > 0.1 * np.max(P_corr)))
av_den = np.array([np.sum(A)for A in int_den_corr])
flat_den = np.transpose(int_den_corr)[-1] / P_corr[-1]
mid_den = np.transpose(int_den_corr)[min_index] / P_corr[min_index]

print np.sum(full_den), np.sum(av_den), np.sum(flat_den), np.sum(mid_den)

fig = plt.figure(2, figsize=(10,5))
plt.plot(Z2, full_den)
plt.plot(Z2, av_den)
#plt.plot(Z2, flat_den)
#plt.plot(Z2, mid_den)
#plt.plot(NZ, [ut.laplace(x, 1, b_est) * (2 * b_est * P_corr[-1]) for x in NZ])
#plt.plot(NZ, [ut.laplace(x, 1, 1 / (2 * P_corr[-1])) for x in NZ])
plt.savefig('/home/fl7g13/Documents/Thesis/Figures/{}_{}_10/{}_{}_{}_{}_{}_{}_{}_int_den.png'.format(model.upper(), csize, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), nframe))
plt.close(2)

fig = plt.figure(2, figsize=(10,5))
plt.plot(N, ut.normalise(P_z))
plt.plot(N, ut.normalise(P_corr))
plt.axis([0, 50 * lslice, 0, 1.0])
#plt.plot(NZ, [ut.laplace(x, 1, b_est) * (2 * b_est * P_corr[-1]) for x in NZ])
#plt.plot(NZ, [ut.laplace(x, 1, 1 / (2 * P_corr[-1])) for x in NZ])
plt.savefig('/home/fl7g13/Documents/Thesis/Figures/{}_{}_10/{}_{}_{}_{}_{}_{}_{}_P_z_nz.png'.format(model.upper(), csize, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), nframe))

fig = plt.figure(4, figsize=(10,5))
plt.plot(Z2, full_den)
plt.plot(Z2, flat_den)
plt.plot(Z2, mid_den)
plt.axis([-10, 10, 0, 0.12])
plt.savefig('/home/fl7g13/Documents/Thesis/Figures/{}_{}_10/{}_{}_{}_{}_{}_{}_{}_den_corr.png'.format(model.upper(), csize, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), nframe))

plt.close('all')

cw_array = np.zeros((4, nslice))

STD = np.sqrt(0.5 * (Delta1 + Delta2))
length = int(STD / lslice) * 12
ZG = np.arange(-lslice*length/2, lslice*length/2 + lslice/2, lslice)
P_den = [ut.gaussian(z, 0, STD) for z in ZG]

for n1, z1 in enumerate(Z1):
	for n2, z2 in enumerate(ZG):
		sys.stdout.write("PERFORMING GAUSSIAN SMOOTHING {0:.1%} COMPLETE \r".format(float(n1 * nslice + n2) / nslice**2) )
		sys.stdout.flush()

		index = int((z1 - z2) / DIM[2] * nslice) % nslice
		try:
			cw_array[0][n1] += av_den[index] * P_den[n2] * lslice
			cw_array[1][n1] += full_den[index] * P_den[n2] * lslice
			cw_array[2][n1] += flat_den[index] * P_den[n2] * lslice
			cw_array[3][n1] += mid_den[index] * P_den[n2] * lslice
		except IndexError: pass


sm_mol_den = [ut.den_func(z, av_den[3 * nslice / 4], 0, DIM[2]/2, param[3], param[4]) for z in Z1]

print np.sum(sm_mol_den), np.sum(cw_array[0]), np.sum(cw_array[1]), np.sum(cw_array[2]), np.sum(cw_array[3])

fig = plt.figure(4, figsize=(10,5))
plt.plot(Z2, cw_array[0])
plt.plot(Z2, cw_array[1])
plt.plot(Z2, cw_array[2])
plt.plot(Z2, cw_array[3])
plt.plot(Z2, sm_mol_den, c='black', linestyle='dashed')
plt.axis([0, DIM[2], 0, 0.05])
plt.savefig('/home/fl7g13/Documents/Thesis/Figures/{}_{}_10/{}_{}_{}_{}_{}_{}_{}_cw_den.png'.format(model.upper(), csize, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), nframe))
plt.close(4)
"""

plot_ism_surface()
