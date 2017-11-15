"""
*************** INTRINSIC SAMPLING METHOD MODULE *******************

Defines coefficients for a fouier series that represents
the periodic surfaces in the xy plane of an air-liquid 
interface. 	

********************************************************************
Created 24/11/16 by Frank Longford

Last modified 22/08/17 by Frank Longford
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

sqrt_2 = np.sqrt(2.)

def combine_intrinsic_surface():

	n_waves = 2 * qm + 1

	tot_coeff = np.zeros((nframe, 2, n_waves**2))
	tot_coeff_recon = np.zeros((nframe, 2, n_waves**2))
	tot_piv = np.zeros((nframe, 2, n0), dtype=int)

	for frame in xrange(nframe):

		file_name = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)
		file_name_recon = '{}_{}_{}_{}_{}_R'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)

		with file('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name), 'r') as infile: 
			tot_coeff[frame][0], tot_coeff[frame][1] = np.load(infile)
		with file('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name_recon), 'r') as infile: 
			tot_coeff_recon[frame][0], tot_coeff_recon[frame][1] = np.load(infile)
		with file('{}/ACOEFF/{}_PIVOTS.npy'.format(directory, file_name), 'r') as infile:
			tot_piv[frame][0], tot_piv[frame][1] = np.load(infile)

	file_name = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), nframe)
	file_name_recon = '{}_{}_{}_{}_{}_R'.format(model.lower(), qm, n0, int(1./phi + 0.5), nframe)

	with file('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name), 'w') as outfile: 
		np.save(outfile, tot_coeff)
	with file('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name_recon), 'w') as outfile: 
		np.save(outfile, tot_coeff_recon)
	with file('{}/ACOEFF/{}_PIVOTS.npy'.format(directory, file_name), 'w') as outfile:
		np.save(outfile, tot_piv)

	for frame in xrange(nframe):
		file_name = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)
		file_name_recon = '{}_{}_{}_{}_{}_R'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)

		os.remove('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name))
		os.remove('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name_recon))
		os.remove('{}/ACOEFF/{}_PIVOTS.npy'.format(directory, file_name))


def intrinsic_surface(directory, xmol, ymol, zmol, model, nsite, nmol, ncube, DIM, qm, n0, phi, psi, vlim, mol_sigma, M, frame, nframe, ow_auv, ow_recon, ow_pos):
	"Creates intrinsic surface of frame." 

	max_r = 1.5 * mol_sigma
	tau = 0.5 * mol_sigma

	file_name = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)
	file_name_recon = '{}_{}_{}_{}_{}_R'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)

	if not os.path.exists("{}/ACOEFF".format(directory)): os.mkdir("{}/ACOEFF".format(directory))
	if not os.path.exists("{}/INTPOS".format(directory)): os.mkdir("{}/INTPOS".format(directory))

	if os.path.exists('{}/ACOEFF/{}_INTCOEFF.txt'.format(directory, file_name)): ut.convert_txt_npy('{}/ACOEFF/{}_INTCOEFF'.format(directory, file_name))
	if os.path.exists('{}/ACOEFF/{}_INTCOEFF.txt'.format(directory, file_name_recon)): ut.convert_txt_npy('{}/ACOEFF/{}_INTCOEFF'.format(directory, file_name_recon))
	if os.path.exists('{}/ACOEFF/{}_PIVOTS.txt'.format(directory, file_name)): ut.convert_txt_npy('{}/ACOEFF/{}_PIVOTS'.format(directory, file_name))

	if not os.path.exists('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name)) or ow_auv:

		sys.stdout.write("PROCESSING {} INTRINSIC SURFACE {}\n".format(directory, frame) )
		sys.stdout.flush()

		auv1, auv2, piv_n1, piv_n2 = build_surface(xmol, ymol, zmol, DIM, nmol, ncube, mol_sigma, qm, n0, phi, vlim, tau, max_r)

		with file('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name), 'w') as outfile:
			np.save(outfile, (auv1, auv2))
		with file('{}/ACOEFF/{}_PIVOTS.npy'.format(directory, file_name), 'w') as outfile:
			np.save(outfile, (piv_n1, piv_n2))

		sys.stdout.write("PROCESSING {}\nINTRINSIC SURFACE RECONSTRUCTION {}\n".format(directory, frame) )
		sys.stdout.flush()

		auv1_recon, auv2_recon = surface_reconstruction(xmol, ymol, zmol, qm, n0, phi, psi, auv1, auv2, piv_n1, piv_n2, DIM)

		with file('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name_recon), 'w') as outfile:
			np.save(outfile, (auv1_recon, auv2_recon))
		
	elif not os.path.exists('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name_recon)) or ow_recon:

		sys.stdout.write("PROCESSING {}\nINTRINSIC SURFACE RECONSTRUCTION {}\n".format(directory, frame) )
		sys.stdout.flush()

		auv1, auv2 = np.load('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name), mmap_mode = 'r')
		piv_n1, piv_n2 = np.load('{}/ACOEFF/{}_PIVOTS.npy'.format(directory, file_name), mmap_mode = 'r')

		auv1_recon, auv2_recon = surface_reconstruction(xmol, ymol, zmol, qm, n0, phi, psi, auv1, auv2, np.array(piv_n1, dtype=int), np.array(piv_n2, dtype=int), DIM)

		with file('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name_recon), 'w') as outfile:
			np.save(outfile, (auv1_recon, auv2_recon))
		
	else: 

		auv1, auv2 = np.load('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name), mmap_mode = 'r')
		auv1_recon, auv2_recon = np.load('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name_recon), mmap_mode = 'r')
		piv_n1, piv_n2 = np.load('{}/ACOEFF/{}_PIVOTS.npy'.format(directory, file_name), mmap_mode = 'r')

	if ow_auv or ow_pos:
		for i, recon in enumerate([False, True]):
			intrinsic_positions_dxdyz(directory, xmol, ymol, zmol, model, frame, nsite, qm, n0, phi, psi, DIM, recon, True)


	return auv1, auv2, auv1_recon, auv2_recon, np.array(piv_n1, dtype=int), np.array(piv_n2, dtype=int)


def build_surface(xmol, ymol, zmol, DIM, nmol, ncube, mol_sigma, qm, n0, phi, vlim, tau, max_r):
	"Create coefficients auv1 and aiv2 for Fourier sum representing intrinsic surface"

	print "\n ------------- BUILDING INTRINSIC SURFACE --------------"

	"""
	xmol, ymol, zmol = x, y, z positions of molecules
	mol_list = index list of molecules eligible to be used as 'pivots' for the fitting routine  
	piv_n = index of pivot molecules
	piv_z = z position of pivot molecules
	new_pivots = list of pivots to be added to piv_n and piv_z
	"""
	tau1 = tau
	tau2 = tau
	mol_list = np.arange(nmol)
	piv_n1 = np.arange(ncube**2)
	piv_z1 = np.zeros(ncube**2)
	piv_n2 = np.arange(ncube**2)
	piv_z2 = np.zeros(ncube**2)
	vapour_list = []
	new_piv1 = []
	new_piv2 = []

	start = time.time()
	
	mat_xmol = np.tile(xmol, (nmol, 1))
	mat_ymol = np.tile(ymol, (nmol, 1))
	mat_zmol = np.tile(zmol, (nmol, 1))

	dr2 = (mat_xmol - np.transpose(mat_xmol))**2 + (mat_ymol - np.transpose(mat_ymol))**2 + (mat_zmol - np.transpose(mat_zmol))**2
	
	"Remove molecules from vapour phase ans assign an initial grid of pivots furthest away from centre of mass"
	print 'Lx = {:5.3f}   Ly = {:5.3f}   qm = {:5d}\nphi = {}   n_piv = {:5d}   vlim = {:5d}   max_r = {:5.3f}'.format(DIM[0], DIM[1], qm, phi, n0, vlim, max_r) 
	print 'Selecting initial {} pivots'.format(ncube**2)

	for n in xrange(nmol):
		vapour = np.count_nonzero(dr2[n] < max_r**2) - 1
		if vapour > vlim:

			indexx = int(xmol[n] * ncube / DIM[0]) % ncube
                        indexy = int(ymol[n] * ncube / DIM[1]) % ncube

			if zmol[n] < piv_z1[ncube*indexx + indexy]:
				piv_n1[ncube*indexx + indexy] = n
				piv_z1[ncube*indexx + indexy] = zmol[n]

			elif zmol[n] > piv_z2[ncube*indexx + indexy]:
				piv_n2[ncube*indexx + indexy] = n
				piv_z2[ncube*indexx + indexy] = zmol[n]

		else: vapour_list.append(n)

	"Update molecular and pivot lists"

	mol_list = [i for i in mol_list if i not in vapour_list]
	mol_list = [i for i in mol_list if i not in piv_n1]
	mol_list = np.array([i for i in mol_list if i not in piv_n2])

	new_piv1 = piv_n1
	new_piv2 = piv_n2

	assert np.sum(np.isin(vapour_list, mol_list)) == 0
	assert np.sum(np.isin(piv_n1, mol_list)) == 0
	assert np.sum(np.isin(piv_n2, mol_list)) == 0

	print piv_n1, piv_n2

	mol_list1 = mol_list[zmol[mol_list] < 0]
	mol_list2 = mol_list[zmol[mol_list] >= 0]

	assert piv_n1 not in mol_list1
	assert piv_n2 not in mol_list2

	n_waves = 2*qm+1

	print 'Initial {} pivots selected: {:10.3f} s'.format(ncube**2, time.time() - start)

	"Form the diagonal xi^2 terms"
	diag = np.zeros(n_waves**2)
	
	for j in xrange(n_waves**2): 
		u = int(j/n_waves)-qm
		v = int(j%n_waves)-qm
		diag[j] += ut.check_uv(u, v) * (u**2 * DIM[1] / DIM[0] + v**2 * DIM[0] / DIM[1])
	diag = np.diagflat(diag)

	diag *= 4 * np.pi**2 * phi
              
	"Create A matrix and b vector for linear algebra equation Ax = b"
	A = np.zeros((2, n_waves**2, n_waves**2))
	b = np.zeros((2, n_waves**2))

	print "{:^77s} | {:^43s} | {:^21s} | {:^21s}".format('TIMINGS (s)', 'PIVOTS', 'TAU', 'INT AREA')
	print ' {:20s}  {:20s}  {:20s}  {:10s} | {:10s} {:10s} {:10s} {:10s} | {:10s} {:10s} | {:10s} {:10s}'.format('Matrix Formation', 'LU Decomposition', 'Pivot selection', 'TOTAL', 'n_piv1', '(new)', 'n_piv2', '(new)', 'surf1', 'surf2', 'surf1', 'surf2')
	print "_" * 170

	building_surface = True
	build_surf1 = True
	build_surf2 = True

	while building_surface:

		start1 = time.time()

		"Update A matrix and b vector"
		temp_A, temp_b = update_A_b(xmol, ymol, zmol, qm, n_waves, new_piv1, new_piv2, DIM)
		A += temp_A
		b += temp_b

		end1 = time.time()

		"Perform LU decomosition to solve Ax = b"
		if len(new_piv1) != 0: auv1 = LU_decomposition(A[0] + diag, b[0])
		if len(new_piv2) != 0: auv2 = LU_decomposition(A[1] + diag, b[1])

		end2 = time.time()

		if len(piv_n1) == n0: 
			build_surf1 = False
			new_piv1 = []
		if len(piv_n2) == n0: 
			build_surf2 = False
			new_piv2 = []

		if build_surf1: zeta_list1 = zeta_list(xmol, ymol, mol_list1, auv1, qm, DIM)
		if build_surf2: zeta_list2 = zeta_list(xmol, ymol, mol_list2, auv2, qm, DIM)

		if build_surf1 or build_surf2:
			finding_pivots = True
			piv_search1 = True
			piv_search2 = True
		else:
			finding_pivots = False
			building_surface = False
			print "ENDING SEARCH"

                while finding_pivots:

			if piv_search1 and build_surf1: mol_list1, new_piv1, piv_n1 = pivot_selection(zmol, mol_sigma, n0, mol_list1, zeta_list1, piv_n1, tau1)
			if piv_search2 and build_surf2: mol_list2, new_piv2, piv_n2 = pivot_selection(zmol, mol_sigma, n0, mol_list2, zeta_list2, piv_n2, tau2)

                        if len(new_piv1) == 0 and len(piv_n1) < n0: tau1 += 0.1 * tau 
			else: piv_search1 = False

                        if len(new_piv2) == 0 and len(piv_n2) < n0: tau2 += 0.1 * tau 
			else: piv_search2 = False

			if piv_search1 or piv_search2: finding_pivots = True
                        else: finding_pivots = False

		end = time.time()
	
		area1 = slice_area(auv1**2, qm, qm, DIM)
		area2 = slice_area(auv2**2, qm, qm, DIM)

		print ' {:20.3f}  {:20.3f}  {:20.3f}  {:10.3f} | {:10d} {:10d} {:10d} {:10d} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f}'.format(end1 - start1, end2 - end1, end - end2, end - start1, len(piv_n1), len(new_piv1), len(piv_n2), len(new_piv2), tau1, tau2, area1, area2)			

	print '\nTOTAL time: {:7.2f} s \n'.format(end - start)

	return auv1, auv2, piv_n1, piv_n2


def zeta_list(xmol, ymol, mol_list, auv, qm, DIM):

	zeta_list = xi(xmol[mol_list], ymol[mol_list], qm, qm, auv, DIM)
   
	return zeta_list


def pivot_selection(zmol, mol_sigma, n0, mol_list, zeta_list, piv_n, tau):

	zeta = np.abs(zmol[mol_list] - zeta_list)
	new_piv = mol_list[zeta <= tau]
	dz_new_piv = zeta[zeta <= tau]

	ut.bubblesort(new_piv, dz_new_piv)

	piv_n = np.append(piv_n, new_piv)
	if len(piv_n) > n0: 
		new_piv = new_piv[:len(piv_n)-n0]
		piv_n = piv_n[:n0] 
	
	mol_list = np.array([i for i in mol_list if i not in new_piv])

	assert np.sum(np.isin(new_piv, mol_list)) == 0

	return mol_list, new_piv, piv_n


def LU_decomposition(A, b):
	lu, piv  = sp.linalg.lu_factor(A)
	auv = sp.linalg.lu_solve((lu, piv), b)
	return auv


def update_A_b(xmol, ymol, zmol, qm, n_waves, new_piv1, new_piv2, DIM):

	A = np.zeros((2, n_waves**2, n_waves**2))
	b = np.zeros((2, n_waves**2))

	fuv1 = np.zeros((n_waves**2, len(new_piv1)))
	fuv2 = np.zeros((n_waves**2, len(new_piv2)))

	for j in xrange(n_waves**2):
		fuv1[j] = function(xmol[new_piv1], int(j/n_waves)-qm, DIM[0]) * function(ymol[new_piv1], int(j%n_waves)-qm, DIM[1])
		b[0][j] += np.sum(zmol[new_piv1] * fuv1[j])
		fuv2[j] = function(xmol[new_piv2], int(j/n_waves)-qm, DIM[0]) * function(ymol[new_piv2], int(j%n_waves)-qm, DIM[1])
		b[1][j] += np.sum(zmol[new_piv2] * fuv2[j])

	A[0] += np.dot(fuv1, np.transpose(fuv1))
	A[1] += np.dot(fuv2, np.transpose(fuv2))

	return A, b


def surface_reconstruction(xmol, ymol, zmol, qm, n0, phi, psi, auv1, auv2, piv_n1, piv_n2, DIM):

	var_lim = 1E-3
	n_waves = 2*qm + 1

	print "PERFORMING SURFACE RESTRUCTURING"
	print 'Lx = {:5.3f}   Ly = {:5.3f}   qm = {:5d}\nphi = {}  psi = {}  n_piv = {:5d}  var_lim = {}'.format(DIM[0], DIM[1], qm, phi, psi, n0, var_lim) 
	print 'Setting up wave product and coefficient matricies'

	orig_psi1 = psi
	orig_psi2 = psi
	psi1 = psi
	psi2 = psi

	start = time.time()	

	"Form the diagonal xi^2 terms"

	fuv1 = np.array([function(xmol[piv_n1], int(j/n_waves)-qm, DIM[0]) * function(ymol[piv_n1], int(j%n_waves)-qm, DIM[1]) for j in xrange(n_waves**2)])
        b1 = np.array([np.sum(zmol[piv_n1] * fuv1[k]) for k in xrange(n_waves**2)])
        
	fuv2 = np.array([function(xmol[piv_n2], int(j/n_waves)-qm, DIM[0]) * function(ymol[piv_n2], int(j%n_waves)-qm, DIM[1]) for j in xrange(n_waves**2)])
	b2 = np.array([np.sum(zmol[piv_n2] * fuv2[k]) for k in xrange(n_waves**2)])

	diag = np.zeros(n_waves**2)
	coeff = np.zeros((n_waves**2, n_waves**2))

	for j in xrange(n_waves**2):
		u1 = int(j/n_waves)-qm
		v1 = int(j%n_waves)-qm
		diag[j] += ut.check_uv(u1, v1) * (phi * (u1**2 * DIM[1] / DIM[0] + v1**2 * DIM[0] / DIM[1]))
		for k in xrange(j+1):
			u2 = int(k/n_waves)-qm
			v2 = int(k%n_waves)-qm
			coeff[j][k] += 16 * np.pi**4 * (u1**2 * u2**2 / DIM[0]**4 + v1**2 * v2**2 / DIM[1]**4 + (u1**2 * v2**2 + u2**2 * v1**2) / (DIM[0]**2 * DIM[1]**2))
			coeff[k][j] = coeff[j][k]

	diag = 4 * np.pi**2 * np.diagflat(diag) 
	ffuv1 = np.dot(fuv1, np.transpose(fuv1))
	ffuv2 = np.dot(fuv2, np.transpose(fuv2))

	end_setup1 = time.time()

	print "{:^74s} | {:^21s} | {:^43s}".format('TIMINGS (s)', 'PSI', 'VAR(H)' )
	print ' {:20s} {:20s} {:20s} {:10s} | {:10s} {:10s} | {:10s} {:10s} {:10s} {:10s}'.format('Matrix Formation', 'LU Decomposition', 'Var Estimation', 'TOTAL', 'surf1', 'surf2', 'surf1', 'piv1', 'surf2', 'piv2')
	print "_" * 165

	H_var1 = ut.H_var_est(auv1**2, qm, qm, DIM)
	H_var2 = ut.H_var_est(auv2**2, qm, qm, DIM)

	auv1_matrix = np.tile(auv1, (n_waves**2, 1))
	H_piv_var1 = np.sum(auv1_matrix * np.transpose(auv1_matrix) * ffuv1 * coeff / n0)
	auv2_matrix = np.tile(auv2, (n_waves**2, 1))
	H_piv_var2 = np.sum(auv2_matrix * np.transpose(auv2_matrix) * ffuv2 * coeff / n0)

	end_setup2 = time.time()
	
	print ' {:20.3f} {:20.3f} {:20.3f} {:10.3f} | {:10.6f} {:10.6f} | {:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format( end_setup1-start, 0, end_setup2-end_setup1, end_setup2-start, 0, 0, H_var1, H_piv_var1, H_var2, H_piv_var2)

	reconstructing = True
	recon_1 = True
	recon_2 = True
	loop1 = 0
	loop2 = 0

	while reconstructing:

		start1 = time.time()
        
		"Update A matrix and b vector"
		A1 = ffuv1 * (1. + coeff * psi1 / n0)
		A2 = ffuv2 * (1. + coeff * psi2 / n0) 

		end1 = time.time()

		"Perform LU decomosition to solve Ax = b"
		if recon_1: auv1_recon = LU_decomposition(A1 + diag, b1)
		if recon_2: auv2_recon = LU_decomposition(A2 + diag, b2)

		end2 = time.time()

		H_var1_recon = ut.H_var_est(auv1_recon**2, qm, qm, DIM)
		H_var2_recon = ut.H_var_est(auv2_recon**2, qm, qm, DIM)

		if recon_1:
			auv1_matrix = np.tile(auv1_recon, (n_waves**2, 1))
			H_piv_var1_recon = np.sum(auv1_matrix * np.transpose(auv1_matrix) * ffuv1 * coeff / n0)
		if recon_2:
			auv2_matrix = np.tile(auv2_recon, (n_waves**2, 1))
			H_piv_var2_recon = np.sum(auv2_matrix * np.transpose(auv2_matrix) * ffuv2 * coeff / n0)

		end3 = time.time()

		print ' {:20.3f} {:20.3f} {:20.3f} {:10.3f} | {:10.6f} {:10.6f} | {:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(end1 - start1, end2 - end1, end3 - end2, end3 - start1, psi1, psi2, H_var1_recon, H_piv_var1_recon, H_var2_recon, H_piv_var2_recon)

		if abs(H_piv_var1_recon - H_var1_recon) <= var_lim: recon_1 = False
		else: 
			psi1 += orig_psi1 * (H_piv_var1_recon - H_var1_recon)
			if abs(H_var1_recon) > 5 * H_var1 or loop1 > 40:
				orig_psi1 *= 0.5 
				psi1 = orig_psi1
				loop1 = 0
			else: loop1 += 1
		if abs(H_piv_var2_recon - H_var2_recon) <= var_lim: recon_2 = False
		else: 
			psi2 += orig_psi2 * (H_piv_var2_recon - H_var2_recon)
			if abs(H_var2_recon) > 5 * H_var2 or loop2 > 40: 
				orig_psi2 *= 0.5 
				psi2 = orig_psi2
				loop2 = 0
			else: loop2 += 1

		if not recon_1 and not recon_2: reconstructing = False

	end = time.time()

	print '\nTOTAL time: {:7.2f} s \n'.format(end - start)

	return auv1_recon, auv2_recon


def function(x, u, Lx):

	if u >= 0: return np.cos(2 * np.pi * u * x / Lx)
	else: return np.sin(2 * np.pi * abs(u) * x / Lx)


def dfunction(x, u, Lx):

	if u >= 0: return - 2 * np.pi * u  / Lx * np.sin(2 * np.pi * u * x  / Lx)
	else: return 2 * np.pi * abs(u) / Lx * np.cos(2 * np.pi * abs(u) * x / Lx)


def ddfunction(x, u, Lx):

	return - 4 * np.pi**2 * u**2 / Lx**2 * function(x, u, Lx)


def xi(x, y, qm, qu, auv, DIM):

	zeta = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			zeta += function(x, u, DIM[0]) * function(y, v, DIM[1]) * auv[j]
	return zeta


def dxyi(x, y, qm, qu, auv, DIM):

	dzx = 0
	dzy = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			dzx += dfunction(x, u, DIM[0]) * function(y, v, DIM[1]) * auv[j]
			dzy += function(x, u, DIM[0]) * dfunction(y, v, DIM[1]) * auv[j]
	return dzx, dzy


def ddxyi(x, y, qm, qu, auv, DIM):


	ddzx = 0
	ddzy = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			ddzx += ddfunction(x, u, DIM[0]) * function(y, v, DIM[1]) * auv[j]
			ddzy += function(x, u, DIM[0]) * ddfunction(y, v, DIM[1]) * auv[j]
	return ddzx, ddzy


def mean_curve_est(x, y, qm, qu, auv, DIM):

	H = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			H += -4 * np.pi**2 * (u**2 / DIM[0]**2 + v**2 / DIM[1]**2) * function(x, u, DIM[0]) * function(y, v, DIM[1]) * auv[j]
	return H


def optimise_ns(directory, model, csize, nmol, nsite, qm, phi, vlim, ncube, DIM, COM, M, mol_sigma, start_ns, end_ns):

	if not os.path.exists('{}/ACOEFF'.format(directory)): os.mkdir('{}/DATA/ACOEFF'.format(directory))

	mol_ex_1 = []
	mol_ex_2 = []

	nframe = 20

	NS = np.arange(start_ns, end_ns, 0.05)
	
	print NS

	for ns in NS:

		n0 = int(DIM[0] * DIM[1] * ns / mol_sigma**2)

		tot_piv_n1 = np.zeros((nframe, n0))
		tot_piv_n2 = np.zeros((nframe, n0))

		for frame in xrange(nframe):
			xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)
			xR, yR, zR = COM[frame]
			auv1, auv2, piv_n1, piv_n2 = intrinsic_surface(directory, model, csize, nsite, nmol, ncube, DIM, COM, qm, n0, phi, vlim, mol_sigma, M, frame, nframe,False, True)

			tot_piv_n1[frame] += piv_n1
			tot_piv_n2[frame] += piv_n2 

		ex_1, ex_2 = mol_exchange(tot_piv_n1, tot_piv_n2, nframe, n0)

		mol_ex_1.append(ex_1)
		mol_ex_2.append(ex_2)

		print ns, n0, ex_1, ex_2

	print NS[np.argmin((np.array(mol_ex_1) + np.array(mol_ex_2)) / 2.)], np.min((np.array(mol_ex_1) + np.array(mol_ex_2)) / 2.)

	#"""
	plt.scatter(NS, (np.array(mol_ex_1) + np.array(mol_ex_2)) / 2.)
	plt.scatter(NS, mol_ex_1, c='g')
	plt.scatter(NS, mol_ex_2, c='r')
	plt.axis([np.min(NS), np.max(NS), 0, np.max(mol_ex_1)])
	plt.show()
	#"""

	return NS[np.argmin((np.array(mol_ex_1) + np.array(mol_ex_2)) / 2.)]


def mol_exchange(piv_1, piv_2, nframe, n0):

	n_1 = 0
	n_2 = 0

	for frame in xrange(nframe-1):

		n_1 += len(set(piv_1[frame]) - set(piv_1[frame+1]))
		n_2 += len(set(piv_2[frame]) - set(piv_2[frame+1]))

	return n_1 / (n0 * float(nframe-1) * 1000), n_2 / (n0 * float(nframe-1) * 1000)


def area_correction(z, auv_2, qm, qu, DIM):

        Axi = 0

        for u in xrange(-qu, qu+1):
                for v in xrange(-qu, qu+1):
                        j = (2 * qm + 1) * (u + qm) + (v + qm)
                        dot_prod = 4 * np.pi**2  * (u**2/DIM[0]**2 + v**2/DIM[1]**2)

			if dot_prod != 0:
                        	f_2 = ut.check_uv(u, v) * auv_2[j] / 4.
                        	Axi += f_2 * dot_prod / (1 + np.sqrt(f_2) * abs(z) * dot_prod)**2 

        return 1 + 0.5*Axi



def slice_area(auv_2, qm, qu, DIM):
	"Obtain the intrinsic surface area"

        Axi = 0.0

	for u in xrange(-qu, qu+1):
                for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			dot_prod = np.pi**2  * (u**2/DIM[0]**2 + v**2/DIM[1]**2)

			if dot_prod != 0:
				f_2 = ut.check_uv(u, v) * auv_2[j]
				Axi += f_2 * dot_prod

        return 1 + 0.5*Axi


def auv_qm(auv, qm, qu):

	auv_qm = np.zeros((2*qu+1)**2)

	for u in xrange(-qu, qu+1):
                for v in xrange(-qu, qu+1):
			j1 = (2 * qm + 1) * (u + qm) + (v + qm)
			j2 = (2 * qu + 1) * (u + qu) + (v + qu)

			auv_qm[j2] = auv[j1] 
	return auv_qm

def auv2_to_f2(auv2, qm):

	f2 = np.zeros((2*qm+1)**2)

	for u in xrange(-qm, qm+1):
                for v in xrange(-qm, qm+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm) 
			f2[j] = auv2[j] * ut.check_uv(u, v) / 4.
	return f2


def auv_xy_correlation(auv_2, qm, qu):

	auv_2[len(auv_2)/2] = 0
	f2 = auv2_to_f2(auv_2, qm)
	
	f2_qm = auv_qm(f2, qm, qu).reshape(((2*qu+1), (2*qu+1)))
	xy_corr = np.fft.fftshift(np.fft.ifftn(f2_qm))
	#xy_corr = np.fft.ifftn(f2_qm)

	return np.abs(xy_corr) * (2*qu+1)**2 / np.sum(f2_qm)


def auv_correlation(auv_t, qm):

      	tot_Gamma = np.zeros((2*qm+1)**2)
	tot_omega = np.zeros((2*qm+1)**2)

	l = len(tot_Gamma)/4
	dtau = 1

        for u in xrange(-qm, qm+1):
                for v in xrange(-qm, qm+1):
                        j = (2 * qm + 1) * (u + qm) + (v + qm)

			ACF_auv = ut.autocorr(auv_t[j]) #* np.mean(auv_t[j]**2)

			try:
				opt, ocov = curve_fit(hydro_func, np.arange(l)*dtau, ACF_auv[:l])

				#print u, v, opt
				"""
				if abs(u) < 5 and abs(v) < 5:
					curve = [hydro_func(t, tot_Gamma[j-1], tot_Gamma[j-1]) for t in np.linspace(0, l*dtau, l*10)]
					plt.figure(0)
					plt.title('{} {}'.format(u, v))
					plt.plot(np.arange(len(auv_t[j])) * dtau, auv_t[j])
					plt.figure(1)
					plt.title('{} {}'.format(u, v))
					plt.plot(np.arange(l)*dtau, ACF_auv[:l])
					plt.plot(np.linspace(0, l*dtau, l*10), curve)
					plt.show()
				#"""			

				tot_Gamma[j] = opt[0]
				tot_omega[j] = opt[1]

			except:

				tot_Gamma[j] = np.nan
				tot_omega[j] = np.nan

				"""
				print ACF_auv[0], np.mean(auv_t[j]**2), np.var(auv_t[j])
				curve = [hydro_func(t, tot_Gamma[j-1], tot_Gamma[j-1]) for t in np.linspace(0, l*5., l*100)]

				plt.figure(0)
				plt.title('{} {}'.format(u, v))
				plt.plot(np.arange(len(auv_t[j])), auv_t[j])
				plt.figure(1)
				plt.title('{} {}'.format(u, v))
				plt.plot(np.arange(l), ACF_auv[:l])
				plt.plot(np.arange(l), ut.autocorr(auv_t[j-1])[:l])
				plt.plot(np.linspace(0, l*5, l*100), curve)
				plt.show()
				"""

        return tot_Gamma, tot_omega


def hydro_func(t, Gamma, omega):

	return np.exp(-Gamma * t) * np.cos(omega * t)


def get_hydro_param(tot_Gamma, tot_omega, qm, qu, DIM, q2_set):

	Gamma_list = []
        Gamma_hist = np.zeros(len(q2_set))
	omega_list = []
        omega_hist = np.zeros(len(q2_set))

        count = np.zeros(len(q2_set))

	for u in xrange(-qu, qu+1):
                for v in xrange(-qu, qu+1):
                        j = (2 * qm + 1) * (u + qm) + (v + qm)
			set_index = np.round(u**2*DIM[1]/DIM[0] + v**2*DIM[0]/DIM[1], 4)	
	
			if set_index != 0:
                                Gamma_list.append(tot_Gamma[j])
                                Gamma_hist[q2_set == set_index] += tot_Gamma[j]
				omega_list.append(tot_omega[j])
                                omega_hist[q2_set == set_index] += tot_omega[j]

                                count[q2_set == set_index] += 1

        for i in xrange(len(q2_set)):
                if count[i] != 0: 
			Gamma_hist[i] *= 1. / count[i]
			omega_hist[i] *= 1. / count[i]
			
	return Gamma_hist, omega_hist


def gamma_q_auv(auv_2, qm, qu, DIM, T, q2_set):

	gamma_list = []
	gamma_hist = np.zeros(len(q2_set))
	gamma_count = np.zeros(len(q2_set))

	dim = np.array(DIM) * 1E-10

	coeff = con.k * 1E3 * T

	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			dot_prod = np.pi**2 * (u**2 * dim[1] / dim[0] + v**2 * dim[0] / dim[1])
			set_index = np.round(u**2*dim[1]/dim[0] + v**2*dim[0]/dim[1], 4)

			if set_index != 0:
				gamma = 1. / (ut.check_uv(u, v) * auv_2[j] * 1E-20 * dot_prod)
				gamma_list.append(gamma)
				gamma_hist[q2_set == set_index] += gamma
				gamma_count[q2_set == set_index] += 1

	for i in xrange(len(q2_set)):
		if gamma_count[i] != 0: gamma_hist[i] *= 1. / gamma_count[i]

	return gamma_hist * coeff#np.array(gamma_list) * coeff#, 


def power_spec_auv(auv_2, qm, qu, DIM, q2_set):

	p_spec_list = []
	p_spec_hist = np.zeros(len(q2_set))
	p_spec_count = np.zeros(len(q2_set))

	dim = np.array(DIM) * 1E-10

	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			set_index = np.round(u**2*dim[1]/dim[0] + v**2*dim[0]/dim[1], 4)

			if set_index != 0:
				p_spec = auv_2[j] * ut.check_uv(u, v) / 4.
				p_spec_list.append(p_spec)
				p_spec_hist[q2_set == set_index] += p_spec
				p_spec_count[q2_set == set_index] += 1

	for i in xrange(len(q2_set)):
		if p_spec_count[i] != 0: p_spec_hist[i] *= 1. / p_spec_count[i]

	return p_spec_hist


def gamma_q_f(f_2, qm, qu, DIM, T, q2_set):

	gamma_list = []
	gamma_hist = np.zeros(len(q2_set))
	gamma_count = np.zeros(len(q2_set))

	DIM = np.array(DIM) * 1E-10
	f_2 *= 1E-20

	coeff = con.k * 1E3 * T / (DIM[0] * DIM[1])

	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			dot_prod = 4 * np.pi**2 * (u**2 / DIM[0]**2 + v**2 / DIM[1]**2)
			set_index = u**2 + v**2

			if abs(u) + abs(v) == 0: pass
			else:
				if u == 0 or v == 0: gamma = 1. / (f_2[j] * dot_prod)
				else: gamma = 1. / (f_2[j] * dot_prod)
				gamma_list.append(gamma)
				gamma_hist[q2_set == set_index] += gamma
				gamma_count[q2_set == set_index] += 1

	for i in xrange(len(q2_set)):
		if gamma_count[i] != 0: gamma_hist[i] *= 1. / gamma_count[i]

	return np.array(gamma_list) * coeff, gamma_hist * coeff


def intrinsic_positions_dxdyz(directory, xmol, ymol, zmol, model, frame, nsite, qm, n0, phi, psi, DIM, recon, ow_all):

	file_name_pos = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1/phi + 0.5), frame)
	file_name_auv = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)
	if recon: 
		file_name_pos = '{}_R'.format(file_name_pos)
		file_name_auv = '{}_R'.format(file_name_auv)

	"""
	if os.path.exists('{}/INTPOS/{}_INTDDXDDY_MOL.npy'.format(directory, file_name_pos)) and not ow_all:
		try:
			with file('{}/INTPOS/{}_INTZ_MOL.npy'.format(directory, file_name_pos), 'r') as infile:
				int_z_mol = np.load(infile)
			with file('{}/INTPOS/{}_INTDXDY_MOL.npy'.format(directory, file_name_pos), 'r') as infile:
				dxdyz_mol = np.load(infile)
			with file('{}/INTPOS/{}_INTDDXDDY_MOL.npy'.format(directory, file_name_pos), 'r') as infile:
				ddxddyz_mol = np.load(infile)
		except:	ow_all = True
	"""

	if not os.path.exists('{}/INTPOS/{}_INTDDXDDY_MOL.npy'.format(directory, file_name_pos)) or ow_all:

		nmol = len(xmol)

		int_z_mol = np.zeros((qm+1, 2, nmol))
		dxdyz_mol = np.zeros((qm+1, 4, nmol)) 
		ddxddyz_mol = np.zeros((qm+1, 4, nmol))

		temp_int_z_mol = np.zeros((2, nmol))
		temp_dxdyz_mol = np.zeros((4, nmol)) 
		temp_ddxddyz_mol = np.zeros((4, nmol))

		auv1, auv2 = np.load('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name_auv), mmap_mode='r')

		for qu in xrange(qm+1):
			sys.stdout.write("PROCESSING {} INTRINSIC POSITIONS AND DXDY {}: qm = {} qu = {}\r".format(directory, frame, qm, qu))
			sys.stdout.flush()

			temp_file_name_pos = '{}_{}_{}_{}_{}_{}'.format(model.lower(), qm, qu, n0, int(1/phi + 0.5), frame)
			if recon: temp_file_name_pos = '{}_R'.format(temp_file_name_pos)

			if os.path.exists('{}/INTPOS/{}_INTZ_AT.txt'.format(directory, temp_file_name_pos)): os.remove('{}/INTPOS/{}_INTZ_AT.txt'.format(directory, temp_file_name_pos))
			if os.path.exists('{}/INTPOS/{}_INTZ_MOL.txt'.format(directory, temp_file_name_pos)): ut.convert_txt_npy('{}/INTPOS/{}_INTZ_MOL'.format(directory, temp_file_name_pos))
			if os.path.exists('{}/INTPOS/{}_INTDXDY_MOL.txt'.format(directory, temp_file_name_pos)): ut.convert_txt_npy('{}/INTPOS/{}_INTDXDY_MOL'.format(directory, temp_file_name_pos))
			if os.path.exists('{}/INTPOS/{}_INTDDXDDY_MOL.txt'.format(directory, temp_file_name_pos)): ut.convert_txt_npy('{}/INTPOS/{}_INTDDXDDY_MOL'.format(directory, temp_file_name_pos))

			try:
				with file('{}/INTPOS/{}_INTZ_MOL.npy'.format(directory, temp_file_name_pos), 'r') as infile:
					temp_int_z_mol += np.load(infile)
				with file('{}/INTPOS/{}_INTDXDY_MOL.npy'.format(directory, temp_file_name_pos), 'r') as infile:
					temp_dxdyz_mol += np.load(infile)
				with file('{}/INTPOS/{}_INTDDXDDY_MOL.npy'.format(directory, temp_file_name_pos), 'r') as infile:
					temp_ddxddyz_mol += np.load(infile)

				os.remove('{}/INTPOS/{}_INTZ_MOL.npy'.format(directory, temp_file_name_pos))
				os.remove('{}/INTPOS/{}_INTDXDY_MOL.npy'.format(directory, temp_file_name_pos))
				os.remove('{}/INTPOS/{}_INTDDXDDY_MOL.npy'.format(directory, temp_file_name_pos))

			except: 
				if qu == 0:
					j = (2 * qm + 1) * qm + qm

					f_x = function(xmol, 0, DIM[0])
					f_y = function(ymol, 0, DIM[1])

					temp_int_z_mol[0] += f_x * f_y * auv1[j]
					temp_int_z_mol[1] += f_x * f_y * auv2[j]

				else:
					for u in [-qu, qu]:
						for v in xrange(-qu, qu+1):
							j = (2 * qm + 1) * (u + qm) + (v + qm)

							f_x = function(xmol, u, DIM[0])
							f_y = function(ymol, v, DIM[1])
							df_dx = dfunction(xmol, u, DIM[0])
							df_dy = dfunction(ymol, v, DIM[1])
							ddf_ddx = ddfunction(xmol, u, DIM[0])
							ddf_ddy = ddfunction(ymol, v, DIM[1])

							temp_int_z_mol[0] += f_x * f_y * auv1[j]
							temp_int_z_mol[1] += f_x * f_y * auv2[j]
							temp_dxdyz_mol[0] += df_dx * f_y * auv1[j]
							temp_dxdyz_mol[1] += f_x * df_dy * auv1[j]
							temp_dxdyz_mol[2] += df_dx * f_y * auv2[j]
							temp_dxdyz_mol[3] += f_x * df_dy * auv2[j]
							temp_ddxddyz_mol[0] += ddf_ddx * f_y * auv1[j]
							temp_ddxddyz_mol[1] += f_x * ddf_ddy * auv1[j]
							temp_ddxddyz_mol[2] += ddf_ddx * f_y * auv2[j]
							temp_ddxddyz_mol[3] += f_x * ddf_ddy * auv2[j]

					for u in xrange(-qu+1, qu):
						for v in [-qu, qu]:
							j = (2 * qm + 1) * (u + qm) + (v + qm)

							f_x = function(xmol, u, DIM[0])
							f_y = function(ymol, v, DIM[1])
							df_dx = dfunction(xmol, u, DIM[0])
							df_dy = dfunction(ymol, v, DIM[1])
							ddf_ddx = ddfunction(xmol, u, DIM[0])
							ddf_ddy = ddfunction(ymol, v, DIM[1])

							temp_int_z_mol[0] += f_x * f_y * auv1[j]
							temp_int_z_mol[1] += f_x * f_y * auv2[j]
							temp_dxdyz_mol[0] += df_dx * f_y * auv1[j]
							temp_dxdyz_mol[1] += f_x * df_dy * auv1[j]
							temp_dxdyz_mol[2] += df_dx * f_y * auv2[j]
							temp_dxdyz_mol[3] += f_x * df_dy * auv2[j]
							temp_ddxddyz_mol[0] += ddf_ddx * f_y * auv1[j]
							temp_ddxddyz_mol[1] += f_x * ddf_ddy * auv1[j]
							temp_ddxddyz_mol[2] += ddf_ddx * f_y * auv2[j]
							temp_ddxddyz_mol[3] += f_x * ddf_ddy * auv2[j]

			int_z_mol[qu] += temp_int_z_mol
			dxdyz_mol[qu] += temp_dxdyz_mol
			ddxddyz_mol[qu] += temp_ddxddyz_mol

		with file('{}/INTPOS/{}_INTZ_MOL.npy'.format(directory, file_name_pos), 'w') as outfile:
			np.save(outfile, (int_z_mol))
		with file('{}/INTPOS/{}_INTDXDY_MOL.npy'.format(directory, file_name_pos), 'w') as outfile:
	       		np.save(outfile, (dxdyz_mol))
		with file('{}/INTPOS/{}_INTDDXDDY_MOL.npy'.format(directory, file_name_pos), 'w') as outfile:
	       		np.save(outfile, (ddxddyz_mol))

	#return int_z_mol, dxdyz_mol, ddxddyz_mol


def intrinsic_local_frame(directory, xat, yat, zat, qm, n0, phi, nmol, frame, model, nsite, eig_vec, recon, ow_local):

	file_name_pos = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)
	if recon: file_name_pos = '{}_R'.format(file_name_pos)

	if not os.path.exists('{}/INTEULER/{}_ODIST.npy'.format(directory, file_name_pos)) or ow_local:

		dxdyz_mol = np.load('{}/INTPOS/{}_INTDXDY_MOL.npy'.format(directory, file_name_pos), mmap_mode='r')

		Odist = np.zeros((qm+1, 2, nmol, 9))

		xat_mol = xat.reshape((nmol, nsite))
		yat_mol = yat.reshape((nmol, nsite))
		zat_mol = zat.reshape((nmol, nsite))

		molecules = np.stack((xat_mol, yat_mol, zat_mol), axis=2)

		O = ut.local_frame_molecule(molecules, model, eig_vec)

		for qu in xrange(qm+1):
			sys.stdout.write("PROCESSING {} ODIST {}: qm = {} qu = {}\r".format(directory, frame, qm, qu) )
			sys.stdout.flush()

			T1 = ut.local_frame_surface(dxdyz_mol[qu][0], dxdyz_mol[qu][1], -1)
			T2 = ut.local_frame_surface(dxdyz_mol[qu][2], dxdyz_mol[qu][3], 1)

			T1inv = np.linalg.inv(T1)
			T2inv = np.linalg.inv(T2)

			R1 = np.matmul(O, T1inv)
			R2 = np.matmul(O, T2inv)

			Odist[qu][0] += R1.reshape((nmol, 9))
			Odist[qu][1] += R2.reshape((nmol, 9))

		with file('{}/INTEULER/{}_ODIST.npy'.format(directory, file_name_pos), 'w') as outfile:
			np.save(outfile, (Odist))


def intrinsic_z_den_corr(directory, zmol, model, qm, n0, phi, psi, frame, nslice, nsite, DIM, recon, ow_count):
	"Saves atom, mol and mass intrinsic profiles of trajectory frame" 

	lslice = DIM[2] / nslice
	nz = 100
        nnz = 100

	file_name_pos = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)
	file_name_count = '{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, n0, int(1./phi + 0.5), frame)	
	file_name_norm = '{}_{}_{}_{}_{}_{}'.format(model.lower(), nz, qm, n0, int(1./phi + 0.5), frame)
	file_name_auv = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)

	if recon:
		file_name_pos = '{}_R'.format(file_name_pos)
		file_name_count = '{}_R'.format(file_name_count)
		file_name_norm = '{}_R'.format(file_name_norm)
		file_name_auv = '{}_R'.format(file_name_auv)	

	try:
		count_corr_array = np.load('{}/INTDEN/{}_COUNTCORR.npy'.format(directory, file_name_count), mmap_mode='r')
		z_nz_array = np.load('{}/INTDEN/{}_N_NZ.npy'.format(directory, file_name_norm), mmap_mode='r')
        except: ow_count = True

	if ow_count:

		auv1, auv2 = np.load('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name_auv), mmap_mode='r')	

		count_corr_array = np.zeros((qm+1, nslice, nnz))
		z_nz_array = np.zeros((qm+1, nz, nnz))	

		nmol = len(zmol)

		try:
			int_z_mol = np.load('{}/INTPOS/{}_INTZ_MOL.npy'.format(directory, file_name_pos), mmap_mode='r')
			dxdyz_mol = np.load('{}/INTPOS/{}_INTDXDY_MOL.npy'.format(directory, file_name_pos), mmap_mode='r')
		except:
			int_z_mol, dxdyz_mol, ddxddyz_mol = intrinsic_positions_dxdyz(directory, xmol, ymol, zmol, model, frame, nsite, qm, n0, phi, psi, DIM, recon, False)

		for qu in xrange(qm+1):
			sys.stdout.write("PROCESSING {} INTRINSIC DENSITY {}: qm = {} qu = {}\r".format(directory, frame, qm, qu) )
			sys.stdout.flush()

			temp_file_name_count = '{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, qu, n0, int(1/phi + 0.5), frame)
			temp_file_name_norm = '{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), nz, qm, qu, n0, int(1/phi + 0.5), frame)

			if recon: 
				temp_file_name_count = '{}_R'.format(temp_file_name_count)
				temp_file_name_norm = '{}_R'.format(temp_file_name_norm)

			try:
				with file('{}/INTDEN/{}_COUNTCORR.npy'.format(directory, temp_file_name_count)) as infile:
					temp_count_corr_array += np.load(infile)
				with file('{}/INTDEN/{}_N_NZ.npy'.format(directory, temp_file_name_norm)) as infile:
                                	temp_z_nz_array += np.load(infile)
				os.remove('{}/INTDEN/{}_COUNTCORR.npy'.format(directory, temp_file_name_count))
				os.remove('{}/INTDEN/{}_N_NZ.npy'.format(directory, temp_file_name_norm))

			except Exception: 

				temp_count_corr_array = np.zeros((nslice, nnz))
				temp_z_nz_array = np.zeros((nz, nnz))

				int_z1 = int_z_mol[qu][0]
				int_z2 = int_z_mol[qu][1]

				z1 = zmol - int_z1
				z2 = -zmol + int_z2

				dzx1 = dxdyz_mol[qu][0]
				dzy1 = dxdyz_mol[qu][1]
				dzx2 = dxdyz_mol[qu][2]
				dzy2 = dxdyz_mol[qu][3]

				index1_mol = np.array((z1 + DIM[2]/2.) * nslice / DIM[2], dtype=int) % nslice
				index2_mol = np.array((z2 + DIM[2]/2.) * nslice / DIM[2], dtype=int) % nslice

				normal1 = abs(ut.unit_vector([-dzx1, -dzy1, np.ones(nmol)])[2])
				normal2 = abs(ut.unit_vector([-dzx2, -dzy2, np.ones(nmol)])[2])

				index1_nz = np.array(normal1 * nnz, dtype=int) % nnz
				index2_nz = np.array(normal2 * nnz, dtype=int) % nnz

				temp_count_corr_array += np.histogram2d(index1_mol, normal1, bins=[nslice, nnz], range=[[0, nslice], [0, nnz]])[0]
				temp_count_corr_array += np.histogram2d(index2_mol, normal2, bins=[nslice, nnz], range=[[0, nslice], [0, nnz]])[0]

				index1_mol = np.array(abs(int_z1 - auv1[len(auv1)/2]) * 2 * nz / (nz*lslice), dtype=int) % nz
				index2_mol = np.array(abs(int_z2 - auv2[len(auv2)/2]) * 2 * nz / (nz*lslice), dtype=int) % nz

				temp_z_nz_array += np.histogram2d(index1_mol, normal1, bins=[nz, nnz], range=[[0, nz], [0, nnz]])[0]
				temp_z_nz_array += np.histogram2d(index2_mol, normal2, bins=[nz, nnz], range=[[0, nz], [0, nnz]])[0]

			count_corr_array[qu] += temp_count_corr_array
			z_nz_array[qu] += temp_z_nz_array

		with file('{}/INTDEN/{}_COUNTCORR.npy'.format(directory, file_name_count), 'w') as outfile:
			np.save(outfile, (count_corr_array))
		with file('{}/INTDEN/{}_N_NZ.npy'.format(directory, file_name_norm), 'w') as outfile:
                        np.save(outfile, (z_nz_array))

	return count_corr_array, z_nz_array


def intrinsic_R_tensors_old(directory, zmol, model, frame, nslice, com, DIM, nsite, qm, n0, phi, psi, recon, ow_R):

	file_name_pos = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1/phi + 0.5), frame)
	file_name_euler = '{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, n0, int(1/phi + 0.5), frame)	
	file_name_auv = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)

	if recon:
		file_name_pos = '{}_R'.format(file_name_pos)
		file_name_euler = '{}_R'.format(file_name_euler)
		file_name_auv = '{}_R'.format(file_name_auv)

	try:
		int_R = np.load('{}/INTEULER/{}_RDIST.npy'.format(directory, file_name_euler), mmap_mode='r')
	except: ow_R = True

	if ow_R:

		nmol = len(zmol)

		int_R = np.zeros((qm+1, nslice, 9))

		with file('{}/INTEULER/{}_ODIST.npy'.format(directory, file_name_pos), 'r') as infile:
			Odist = np.load(infile)
		with file('{}/INTPOS/{}_INTZ_MOL.npy'.format(directory, file_name_pos), 'r') as infile:
			int_z_mol = np.load(infile)
	
		for qu in xrange(qm+1):
			#sys.stdout.write("PROCESSING {} RDIST {}: qm = {} qu = {}\r".format(directory, frame, qm, qu) )
			#sys.stdout.flush()

			temp_int_R = np.zeros((nslice, 9))

			temp_file_name_euler = '{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, qu, n0, int(1/phi + 0.5), frame)	
			temp_file_name_pos = '{}_{}_{}_{}_{}_{}'.format(model.lower(), qm, qu, n0, int(1/phi + 0.5), frame)

			if recon: 
				temp_file_name_euler += '_R'
				temp_file_name_pos += '_R'

			try:
				with file('{}/INTEULER/{}_ODIST.npy'.format(directory, temp_file_name_euler), 'r') as infile:
					temp_int_R += np.load(infile)
				os.remove('{}/INTEULER/{}_ODIST.npy'.format(directory, temp_file_name_euler))

			except:
				int_z1 = int_z_mol[qu][0]
				int_z2 = int_z_mol[qu][1]

				z1 = zmol - int_z1
				z2 = -zmol + int_z2

				index1_mol = np.array((z1 + DIM[2]/2.) * nslice / DIM[2], dtype=int) % nslice
				index2_mol = np.array((z2 + DIM[2]/2.) * nslice / DIM[2], dtype=int) % nslice

				for j in xrange(nmol):
	
					temp_int_R[index1_mol[j]] += Odist[qu][0][j]**2 
					temp_int_R[index2_mol[j]] += Odist[qu][1][j]**2

			int_R[qu] += temp_int_R

		with file('{}/INTEULER/{}_RDIST.npy'.format(directory, file_name_euler), 'w') as outfile:
			np.save(outfile, int_R)

	return int_R


def intrinsic_R_tensors(directory, zmol, model, frame, nslice, com, DIM, nsite, qm, n0, phi, psi, recon):

	file_name_pos = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1/phi + 0.5), frame)
	file_name_euler = '{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, n0, int(1/phi + 0.5), frame)	
	file_name_auv = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)

	if recon:
		file_name_pos += '_R'
		file_name_euler += '_R'
		file_name_auv += '_R'

	nmol = len(zmol)

	int_R = np.zeros((qm+1, 9, nslice))

	Odist = np.load('{}/INTEULER/{}_ODIST.npy'.format(directory, file_name_pos), mmap_mode='r')
	int_z_mol = np.load('{}/INTPOS/{}_INTZ_MOL.npy'.format(directory, file_name_pos), mmap_mode='r')

	try: os.remove('{}/INTEULER/{}_RDIST.npy'.format(directory, file_name_euler))
	except: pass

	Odist = np.moveaxis(Odist, -1, -2)**2

	for qu in xrange(qm+1):
		temp_file_name_euler = '{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, qu, n0, int(1/phi + 0.5), frame)
		try: os.remove('{}/INTEULER/{}_ODIST.npy'.format(directory, temp_file_name_euler))
		except: pass
		#sys.stdout.write("PROCESSING {} RDIST {}: qm = {} qu = {}\r".format(directory, frame, qm, qu) )
		#sys.stdout.flush()

		int_z1 = int_z_mol[qu][0]
		int_z2 = int_z_mol[qu][1]

		z1 = zmol - int_z1
		z2 = -zmol + int_z2

		index1_mol = np.array((z1 + DIM[2]/2.) * nslice / DIM[2], dtype=int) % nslice
		index2_mol = np.array((z2 + DIM[2]/2.) * nslice / DIM[2], dtype=int) % nslice

		for j in xrange(9): 
			int_R[qu][j] += np.histogram(index1_mol, bins=nslice, weights=Odist[qu][0][j], range=[0, nslice])[0]
			int_R[qu][j] += np.histogram(index2_mol, bins=nslice, weights=Odist[qu][1][j], range=[0, nslice])[0]


	return np.moveaxis(int_R, -1, -2)


def intrinsic_mol_angles(directory, zmol, model, frame, nslice, npi, nmol, DIM, nsite, qm, n0, phi, psi, recon, ow_angle):

	dpi = np.pi / npi

	file_name_euler = '{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, n0, int(1/phi + 0.5), frame)
	file_name_pos = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1/phi + 0.5), frame)
	file_name_coeff = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)
	file_name_pangle = '{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, npi, qm, n0, int(1/phi + 0.5), frame)

	if recon: 
		file_name_euler += '_R'
		file_name_pos += '_R'
		file_name_coeff += '_R'

	try:
		int_theta, int_phi, int_varphi = np.load('{}/INTEULER/{}_INTANGLE.npy'.format(directory, file_name_pos), mmap_mode='r')
	except: ow_angle = True

	if ow_angle:

		int_P_z_theta_phi = np.zeros((qm+1, nslice, npi, npi*2))

		int_theta = np.zeros((qm+1, 2, nmol))
		int_phi = np.zeros((qm+1, 2, nmol))
		int_varphi = np.zeros((qm+1, 2, nmol))

		Odist = np.load('{}/INTEULER/{}_ODIST.npy'.format(directory, file_name_pos), mmap_mode='r')
		int_z_mol = np.load('{}/INTPOS/{}_INTZ_MOL.npy'.format(directory, file_name_pos), mmap_mode='r')	

		for qu in xrange(qm+1):
	
			temp_file_name_euler = '{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, qu, n0, int(1/phi + 0.5), frame)	
			temp_file_name_pos = '{}_{}_{}_{}_{}_{}'.format(model.lower(), qm, qu, n0, int(1/phi + 0.5), frame)
			temp_file_name_coeff = '{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1./phi + 0.5), frame)

			if recon: 
				temp_file_name_euler += '_R'
				temp_file_name_pos += '_R'
				temp_file_name_coeff += '_R'

			try:
				with file('{}/INTEULER/{}_ANGLE1.npy'.format(directory, temp_file_name_pos), 'r') as infile:
					zeta_array1, int_theta[qu][0], int_phi[qu][0], int_varphi[qu][0] = np.load(infile)
				with file('{}/INTEULER/{}_ANGLE2.npy'.format(directory, temp_file_name_pos), 'r') as infile:
					zeta_array2, int_theta[qu][1], int_phi[qu][1], int_varphi[qu][1] = np.load(infile)

				os.remove('{}/INTEULER/{}_ANGLE1.npy'.format(directory, temp_file_name_pos))
				os.remove('{}/INTEULER/{}_ANGLE2.npy'.format(directory, temp_file_name_pos))

			except Exception:
				sys.stdout.write("PROCESSING {} INTRINSIC ANGLES {}: qm = {} qu = {} \r".format(directory, frame, qm, qu) )
				sys.stdout.flush()

				int_z1 = int_z_mol[qu][0]
				int_z2 = int_z_mol[qu][1]

				zeta_array1 = zmol - int_z1
				zeta_array2 = -zmol + int_z2

				R1 = np.moveaxis(Odist[qu][0], 0, 1)
				R2 = np.moveaxis(Odist[qu][1], 0, 1)

				int_theta[qu][0] = np.arccos(R1[8])
				int_phi[qu][0] = (np.arctan(-R1[6] / R1[7]))
				int_varphi[qu][0] = (np.arctan(R1[2] / R1[5]))

				int_theta[qu][1] = np.arccos(R2[8])
				int_phi[qu][1] = (np.arctan(-R2[6] / R2[7]))
				int_varphi[qu][1] = (np.arctan(R2[2] / R2[5]))


			index_z = np.array((zeta_array1 + DIM[2]/2) * nslice / DIM[2], dtype=int) % nslice
			index_theta = np.array(int_theta[qu][0] /dpi, dtype=int) 
			index_phi = np.array((int_phi[qu][0] + np.pi / 2.) /dpi, dtype=int)

			int_P_z_theta_phi[qu] += np.histogramdd((index_z, index_theta, index_phi), bins=[nslice, npi, npi*2], range=[[0, nslice], [0, npi], [0, npi*2]])[0]

			index_z = np.array((zeta_array2 + DIM[2]/2) * nslice / DIM[2], dtype=int) % nslice
			index_theta = np.array(int_theta[qu][1] /dpi, dtype=int) 
			index_phi = np.array((int_phi[qu][1] + np.pi / 2.) /dpi, dtype=int)

			int_P_z_theta_phi[qu] += np.histogramdd((index_z, index_theta, index_phi), bins=[nslice, npi, npi*2], range=[[0, nslice], [0, npi], [0, npi*2]])[0]

		with file('{}/INTEULER/{}_INTANGLE.npy'.format(directory, file_name_pos), 'w') as outfile:
			np.save(outfile, (int_theta, int_phi, int_varphi))
		with file('{}/INTEULER/{}_INTPANGLE.npy'.format(directory, file_name_pangle), 'w') as outfile:
			np.save(outfile, (int_P_z_theta_phi))

	return int_P_z_theta_phi


def intrinsic_angle_dist(nslice, qm, npi, int_P_z_theta_phi):

	print "BUILDING ANGLE DISTRIBUTIONS"

	dpi = np.pi / npi

	print ""
	print "NORMALISING GRID"

	for qu in xrange(qm+1):
		for index1 in xrange(nslice): 
			if np.sum(int_P_z_theta_phi[qu][index1]) != 0: int_P_z_theta_phi[qu][index1] = int_P_z_theta_phi[qu][index1] / np.sum(int_P_z_theta_phi[qu][index1])

	int_P_z_phi_theta = np.moveaxis(int_P_z_theta_phi, -2, -1)

	X_theta = np.arange(0, np.pi, dpi)
	X_phi = np.arange(-np.pi / 2, np.pi / 2, dpi)

	int_av_theta = np.zeros((qm+1, nslice))
        int_av_phi = np.zeros((qm+1, nslice))
	int_P1 = np.zeros((qm+1, nslice))
	int_P2 = np.zeros((qm+1, nslice))

	print "BUILDING AVERAGE ANGLE PROFILES"

	for qu in xrange(qm+1):
		for index1 in xrange(nslice):
			sys.stdout.write("PROCESSING AVERAGE ANGLE PROFILES {} out of {} slices\r".format(index1, nslice) )
			sys.stdout.flush() 

			for index2 in xrange(npi):
				int_av_theta[qu][index1] += np.sum(int_P_z_theta_phi[qu][index1][index2]) * X_theta[index2] 
				int_P1[qu][index1] += np.sum(int_P_z_theta_phi[qu][index1][index2]) * np.cos(X_theta[index2])
				int_P2[qu][index1] += np.sum(int_P_z_theta_phi[qu][index1][index2]) * 0.5 * (3 * np.cos(X_theta[index2])**2 - 1)

				int_av_phi[qu][index1] += np.sum(int_P_z_phi_theta[qu][index1][index2]) * (X_phi[index2]) 

			if int_av_theta[qu][index1] == 0: 
				int_av_theta[qu][index1] += np.pi / 2.
				int_av_phi[qu][index1] += np.pi / 4.

	a_dist = (int_av_theta, int_av_phi, int_P1, int_P2)
	
	return a_dist


def intrinsic_polarisability(nslice, qm, eig_val, count_int_O, av_int_O):

	int_axx = np.zeros((qm+1, nslice))
	int_azz = np.zeros((qm+1, nslice))

	for qu in xrange(qm+1):
		for n in xrange(nslice):
			if count_int_O[qu][n] != 0:
				av_int_O[qu][n] *= 1./ count_int_O[qu][n]
				for j in xrange(3):
					int_axx[qu][n] += eig_val[j] * 0.5 * (av_int_O[qu][n][j] + av_int_O[qu][n][j+3]) 
					int_azz[qu][n] += eig_val[j] * av_int_O[qu][n][j+6] 
			else: 					
				int_axx[qu][n] = np.mean(eig_val)					
				int_azz[qu][n] = np.mean(eig_val)

	polar = (int_axx, int_azz)

	return polar


def cw_gamma_1(q, gamma, kappa): return gamma + kappa * q**2


def cw_gamma_2(q, gamma, kappa0, l0): return gamma + q**2 * (kappa0 + l0 * np.log(q))


def cw_gamma_dft(q, gamma, kappa, eta0, eta1): return gamma + eta0 * q + kappa * q**2 + eta1 * q**3


def cw_gamma_sk(q, gamma, w0, r0, dp): return gamma + np.pi/32 * w0 * r0**6 * dp**2 * q**2 * (np.log(q * r0 / 2.) - (3./4 * 0.5772))


def intrinsic_profile(directory, model, csize, ntraj, nframe, natom, nmol, nsite, AT, M, a_type, mol_sigma, com, DIM, nslice, ncube, qm, QM, n0, phi, npi, vlim, ow_profile, ow_auv, ow_recon, ow_pos, ow_local, ow_dist, ow_count, ow_angle, ow_polar):

	if model.lower() == 'argon': T = 85
	else: T = 298

	lslice = DIM[2] / nslice
	Aslice = DIM[0]*DIM[1]
	Vslice = DIM[0]*DIM[1]*lslice
	Acm = 1E-8
	ur = 1
	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)
	NZ = np.linspace(0, 1, 100)
	n_waves = 2 * qm + 1
	psi = phi * DIM[0] * DIM[1]

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	eig_val, eig_vec = ut.get_polar_constants(model, a_type)

	av_auv1 = np.zeros((2, nframe))
	av_auv2 = np.zeros((2, nframe))

	av_auv1_2 = np.zeros((2, n_waves**2))
	av_auv2_2 = np.zeros((2, n_waves**2))

	av_auvU_2 = np.zeros((2, n_waves**2))
        av_auvP_2 = np.zeros((2, n_waves**2))

	tot_auv1 = np.zeros((2, nframe, n_waves**2))
	tot_auv2 = np.zeros((2, nframe, n_waves**2))

	file_name_die = ['{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), a_type, nslice, qm, n0, int(1/phi + 0.5), nframe), 
		    	 '{}_{}_{}_{}_{}_{}_{}_R'.format(model.lower(), a_type, nslice, qm, n0, int(1/phi + 0.5), nframe)]
	file_name_den = ['{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, n0, int(1/phi + 0.5), nframe), 
		         '{}_{}_{}_{}_{}_{}_R'.format(model.lower(), nslice, qm, n0, int(1/phi + 0.5), nframe)]
	file_name_hydro = ['{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1/phi + 0.5), nframe), 
		     	'{}_{}_{}_{}_{}_R'.format(model.lower(), qm, n0, int(1/phi + 0.5), nframe)]

	file_check = np.all([os.path.exists('{}/INTDEN/{}_EFF_DEN.npy'.format(directory, file_name_den[1])),
			os.path.exists('{}/INTDIELEC/{}_DIE.npy'.format(directory, file_name_die[1])),
			os.path.exists('{}/INTDIELEC/{}_CWDIE.npy'.format(directory, file_name_die[1])),
			os.path.exists('{}/INTDIELEC/{}_ELLIP_NO.npy'.format(directory, file_name_die[1]))])
			

	if not file_check or ow_auv or ow_pos or ow_local:

		print "IMPORTING GLOBAL POSITION DISTRIBUTIONS\n"

		xat, yat, zat = ut.read_atom_positions(directory, model, csize, ntraj, nframe, com)
		xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, ntraj, nframe, com)
		COM = ut.read_com_positions(directory, model, csize, ntraj, nframe, com)
	
		for frame in xrange(nframe):
			sys.stdout.write("PROCESSING INTRINSIC SURFACE PROFILES {} out of {} frames\r".format(frame, nframe) )
			sys.stdout.flush()

			tot_auv1[0][frame], tot_auv2[0][frame], tot_auv1[1][frame],tot_auv2[1][frame], piv_n1, piv_n2 = intrinsic_surface(directory, xmol[frame], ymol[frame], zmol[frame]-COM[frame][2], model, nsite, nmol, ncube, DIM, qm, n0, phi, psi, vlim, mol_sigma, M, frame, nframe, ow_auv, ow_recon, ow_pos)

			for i, recon in enumerate([False, True]):
				intrinsic_positions_dxdyz(directory, xmol[frame], ymol[frame], zmol[frame]-COM[frame][2], model, frame, nsite, qm, n0, phi, psi, DIM, recon, ow_pos)
				intrinsic_local_frame(directory, xat[frame], yat[frame], zat[frame], qm, n0, phi, nmol, frame, model, nsite, eig_vec, recon, ow_local)	

				av_auv1_2[i] += tot_auv1[i][frame]**2 / nframe
				av_auv2_2[i] += tot_auv2[i][frame]**2 / nframe

				av_auv1[i][frame] = tot_auv1[i][frame][n_waves**2/2]
				av_auv2[i][frame] = tot_auv2[i][frame][n_waves**2/2]

				av_auvU_2[i] += (tot_auv1[i][frame] + tot_auv2[i][frame])**2/ (4. * nframe)
				av_auvP_2[i] += (tot_auv1[i][frame] - tot_auv2[i][frame])**2/ (4. * nframe)
		
		del xat, yat, zat, xmol, ymol, zmol, COM
		gc.collect()

	else:
		for frame in xrange(nframe):
			sys.stdout.write("PROCESSING INTRINSIC SURFACE PROFILES {} out of {} frames\r".format(frame, nframe) )
			sys.stdout.flush()

			file_name_auv = ['{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1/phi + 0.5), frame), 
		     			'{}_{}_{}_{}_{}_R'.format(model.lower(), qm, n0, int(1/phi + 0.5), frame)] 

			for i, recon in enumerate([False, True]):
		
				tot_auv1[i][frame], tot_auv2[i][frame] = np.load('{}/ACOEFF/{}_INTCOEFF.npy'.format(directory, file_name_auv[i]), mmap_mode = 'r')

				av_auv1_2[i] += tot_auv1[i][frame]**2 / nframe
				av_auv2_2[i] += tot_auv2[i][frame]**2 / nframe

				av_auv1[i][frame] = tot_auv1[i][frame][n_waves**2/2]
				av_auv2[i][frame] = tot_auv2[i][frame][n_waves**2/2]

				av_auvU_2[i] += (tot_auv1[i][frame] + tot_auv2[i][frame])**2/ (4. * nframe)
				av_auvP_2[i] += (tot_auv1[i][frame] - tot_auv2[i][frame])**2/ (4. * nframe)


	print "\n"

	AU = np.zeros((2, qm+1))
	AP = np.zeros((2, qm+1))
	ACU = np.zeros((2, qm+1))

	Q_set = []

	cw_gammaU = [[], []]
	cw_gammaP = [[], []]
	cw_gammaCU = [[], []]

	av_auvCU_2 = np.array([av_auvP_2[0] - av_auvU_2[0], av_auvP_2[1] - av_auvU_2[1]])

	tot_auv1 = np.array([np.transpose(tot_auv1[0]), np.transpose(tot_auv2[1])])
	tot_auv2 = np.array([np.transpose(tot_auv2[0]), np.transpose(tot_auv2[1])])

	"""
	for r, recon in enumerate([False, True]):
		if not os.path.exists('{}/INTTHERMO/{}_HYDRO.npy'.format(directory, file_name_hydro[r])) or False:
			tot_Gamma1, tot_omega1 = auv_correlation(tot_auv1[r], qm)
			tot_Gamma2, tot_omega2 = auv_correlation(tot_auv2[r], qm)

			with file('{}/INTTHERMO/{}_HYDRO.npy'.format(directory, file_name_hydro[r]), 'w') as outfile:
		                np.save(outfile, (tot_Gamma1, tot_omega1, tot_Gamma2, tot_omega2))
	
	with file('{}/INTTHERMO/{}_HYDRO.npy'.format(directory, file_name_hydro[0]), 'r') as infile:
		tot_Gamma1, tot_omega1, tot_Gamma2, tot_omega2 = np.load(infile)
	with file('{}/INTTHERMO/{}_HYDRO.npy'.format(directory, file_name_hydro[1]), 'r') as infile:
		tot_Gamma1_recon, tot_omega1_recon, tot_Gamma2_recon, tot_omega2_recon = np.load(infile)
	"""
	#"""
	for qu in xrange(qm+1):
		for r, recon in enumerate([False, True]):
			temp_file_name_die = ['{}_{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), a_type, nslice, qm, qu, n0, int(1/phi + 0.5), nframe), 
			    	 	'{}_{}_{}_{}_{}_{}_{}_{}_R'.format(model.lower(), a_type, nslice, qm, qu, n0, int(1/phi + 0.5), nframe)]
			temp_file_name_den = ['{}_{}_{}_{}_{}_{}_{}'.format(model.lower(), nslice, qm, qu, n0, int(1/phi + 0.5), nframe), 
					 '{}_{}_{}_{}_{}_{}_{}_R'.format(model.lower(), nslice, qm, qu, n0, int(1/phi + 0.5), nframe)]
			if os.path.exists('{}/INTDEN/{}_MOL_DEN.npy'.format(directory, temp_file_name_den[r])):
				os.remove('{}/INTDEN/{}_MOL_DEN.npy'.format(directory, temp_file_name_den[r]))
				os.remove('{}/INTDEN/{}_MOL_DEN_CORR.npy'.format(directory, temp_file_name_den[r]))
				os.remove('{}/INTEULER/{}_INT_EUL.npy'.format(directory, temp_file_name_die[r]))
	#"""

	if not file_check or ow_profile:

		print "ENTERING PROFILE LOOP {} {}\n".format(file_check, ow_profile)

		for r, recon in enumerate([False, True]):

			file_check = np.all([os.path.exists('{}/INTDEN/{}_MOL_DEN.npy'.format(directory, file_name_den[r])),
				os.path.exists('{}/INTDEN/{}_MOL_DEN_CORR.npy'.format(directory, file_name_den[r])),
				os.path.exists('{}/INTEULER/{}_INT_EUL.npy'.format(directory, file_name_die[r]))])

			if not file_check or ow_dist: 

				_, _, zmol = ut.read_mol_positions(directory, model, csize, ntraj, nframe, com)
				COM = ut.read_com_positions(directory, model, csize, ntraj, nframe, com)

				av_den_corr_matrix = np.zeros((qm+1, nslice, 100))
				av_z_nz_matrix = np.zeros((qm+1, 100, 100))	

				#count_int_R = np.zeros((qm+1, nslice))
				av_int_R = np.zeros((qm+1, nslice, 9))

				int_P_z_theta_phi = np.zeros((qm+1, nslice, npi, npi*2))

				for frame in xrange(nframe):
					sys.stdout.write("PROCESSING INTRINSIC DISTRIBUTIONS {} out of {} frames\r".format(frame, nframe) )
					sys.stdout.flush() 

					int_count_corr_array, int_count_z_nz = intrinsic_z_den_corr(directory, zmol[frame]-COM[frame][2], model, qm, n0, phi, psi, frame, nslice, nsite, DIM, recon, ow_count)
					av_den_corr_matrix += int_count_corr_array
					av_z_nz_matrix += int_count_z_nz

					if model.upper() != 'ARGON':

						#temp_int_P_z_theta_phi = intrinsic_mol_angles(directory, zmol[frame]-COM[frame][2], model, frame, nslice, npi, nmol, DIM, nsite, qm, n0, phi, psi, recon, ow_angle)
						#int_P_z_theta_phi += temp_int_P_z_theta_phi

						temp_int_R = intrinsic_R_tensors(directory, zmol[frame]-COM[frame][2], model, frame, nslice, com, DIM, nsite, qm, n0, phi, psi, recon)
						av_int_R += temp_int_R
	
				N = np.linspace(0, 50 * lslice, 100)

				P_z_nz = np.array([matrix / np.sum(matrix) for matrix in av_z_nz_matrix]) * 2 * 0.01 * lslice
				P_den_corr_matrix = np.array([[A / np.sum(A) for A in B] for B in av_den_corr_matrix])
				P_corr = np.array([[A / np.sum(A) for A in np.transpose(B)] for B in P_den_corr_matrix])

				int_den_corr = av_den_corr_matrix / (2 * nframe * Vslice)
				mol_int_den = np.sum(int_den_corr, axis=2)
				count_int_R = np.sum(av_den_corr_matrix, axis=2)


				with file('{}/INTDEN/{}_MOL_DEN.npy'.format(directory, file_name_den[r]), 'w') as outfile:
					np.save(outfile, mol_int_den)
				with file('{}/INTDEN/{}_MOL_DEN_CORR.npy'.format(directory, file_name_den[r]), 'w') as outfile:
					np.save(outfile, int_den_corr)

				if model.upper() != 'ARGON':

					int_axx, int_azz = intrinsic_polarisability(nslice, qm, eig_val, count_int_R, av_int_R)
					int_av_theta = np.zeros(nslice)
					int_av_phi = np.zeros(nslice)
					int_P1 = np.zeros(nslice)
					int_P2 = np.zeros(nslice)
					#int_av_theta, int_av_phi, int_P1, int_P2 = intrinsic_angle_dist(nslice, qm, npi, int_P_z_theta_phi)

					with file('{}/INTEULER/{}_INT_EUL.npy'.format(directory, file_name_die[r]), 'w') as outfile:
						np.save(outfile, (int_axx, int_azz, int_av_theta, int_av_phi, int_P1, int_P2))

			else:
				mol_int_den = np.load('{}/INTDEN/{}_MOL_DEN.npy'.format(directory, file_name_den[r]), mmap_mode='r')

				if model.upper() == 'ARGON':
					int_axx = np.ones((qm+1, nslice)) * eig_val
					int_azz = np.ones((qm+1, nslice)) * eig_val
				else:
					int_axx, int_azz, int_av_theta, int_av_phi, int_P1, int_P2 = np.load('{}/INTEULER/{}_INT_EUL.npy'.format(directory, file_name_die[r]), mmap_mode='r')

			eff_den = np.zeros((qm+1, nslice))
			int_die = np.zeros((qm+1, 2, nslice))
			cw_die = np.zeros((qm+1, 8, nslice))
			ellip_no = np.zeros((qm+1, 4, nslice))	

			for qu in xrange(1, qm+1):

				q_set = []
				q2_set = []

				for u in xrange(-qu, qu):
					for v in xrange(-qu, qu):
						q = 4 * np.pi**2 * (u**2 / DIM[0]**2 + v**2/DIM[1]**2)
						q2 = u**2 * DIM[1]/DIM[0] + v**2 * DIM[0]/DIM[1]

						if q2 not in q2_set:
							q_set.append(q)
							q2_set.append(np.round(q2, 4))

				q_set = np.sqrt(np.sort(q_set, axis=None))
				q2_set = np.sort(q2_set, axis=None)
				Q_set.append(q_set)

				AU[r][qu] = (slice_area(av_auvU_2[r], qm, qu, DIM))
				AP[r][qu] = (slice_area(av_auvP_2[r], qm, qu, DIM))
				ACU[r][qu] = (slice_area(av_auvCU_2[r], qm, qu, DIM))

				cw_gammaU[r].append(gamma_q_auv(av_auvU_2[r]*2, qm, qu, DIM, T, q2_set))
				cw_gammaP[r].append(gamma_q_auv(av_auvP_2[r]*2, qm, qu, DIM, T, q2_set))
				cw_gammaCU[r].append(gamma_q_auv(av_auvCU_2[r], qm, qu, DIM, T, q2_set))
	
				#Gamma_hist1, omega_hist1 = get_hydro_param(tot_Gamma1, tot_omega1, qm, qu, DIM, q2_set)
				#Gamma_hist2, omega_hist2 = get_hydro_param(tot_Gamma2, tot_omega2, qm, qu, DIM, q2_set)
				#Gamma_hist1_recon, omega_hist1_recon = get_hydro_param(tot_Gamma1_recon, tot_omega1_recon, qm, qu, DIM, q2_set)
				#Gamma_hist2_recon, omega_hist2_recon = get_hydro_param(tot_Gamma2_recon, tot_omega2_recon, qm, qu, DIM, q2_set)

				if qu == qm:
					file_name_gamma = ['{}_{}_{}_{}_{}'.format(model.lower(), qm, n0, int(1/phi + 0.5), nframe),
							'{}_{}_{}_{}_{}_R'.format(model.lower(), qm, n0, int(1/phi + 0.5), nframe)]

					with file('{}/INTTHERMO/{}_GAMMA.npy'.format(directory, file_name_gamma[r]), 'w') as outfile:
						np.save(outfile, (Q_set[-1], cw_gammaU[r][-1], cw_gammaP[r][-1], cw_gammaCU[r][-1]))


				Delta1 = (ut.sum_auv_2(av_auv1_2[r], qm, qu) - np.mean(av_auv1[r])**2)
				Delta2 = (ut.sum_auv_2(av_auv2_2[r], qm, qu) - np.mean(av_auv2[r])**2)

				print Delta1, Delta2

				rho_axx =  np.array([mol_int_den[qu][n] * int_axx[qu][n] for n in range(nslice)])
				rho_azz =  np.array([mol_int_den[qu][n] * int_azz[qu][n] for n in range(nslice)])

				int_exx = np.array([(1 + 8 * np.pi / 3. * rho_axx[n]) / (1 - 4 * np.pi / 3. * rho_axx[n]) for n in range(nslice)])
				int_ezz = np.array([(1 + 8 * np.pi / 3. * rho_azz[n]) / (1 - 4 * np.pi / 3. * rho_azz[n]) for n in range(nslice)])

				int_no = np.sqrt(ur * int_exx)
				int_ni = np.sqrt(ur * int_ezz)

				centres = np.ones(9) * (np.mean(av_auv1[r]) - np.mean(av_auv2[r]))/2.
				deltas = np.ones(9) * 0.5 * (Delta1 + Delta2)

				arrays = [mol_int_den[qu], int_axx[qu], int_azz[qu], rho_axx, rho_azz, int_exx, int_ezz, int_no, int_ni]

				cw_arrays = ut.gaussian_smoothing(arrays, centres, deltas, DIM, nslice)

				cw_exx1 = np.array([(1 + 8 * np.pi / 3. * cw_arrays[0][n] * cw_arrays[1][n]) / (1 - 4 * np.pi / 3. * cw_arrays[0][n] * cw_arrays[1][n]) for n in range(nslice)])
				cw_ezz1 = np.array([(1 + 8 * np.pi / 3. * cw_arrays[0][n] * cw_arrays[2][n]) / (1 - 4 * np.pi / 3. * cw_arrays[0][n] * cw_arrays[2][n]) for n in range(nslice)])

				cw_exx2 = np.array([(1 + 8 * np.pi / 3. * cw_arrays[3][n]) / (1 - 4 * np.pi / 3. * cw_arrays[3][n]) for n in range(nslice)])
				cw_ezz2 = np.array([(1 + 8 * np.pi / 3. * cw_arrays[4][n]) / (1 - 4 * np.pi / 3. * cw_arrays[4][n]) for n in range(nslice)])

				print '\n'
				print "WRITING TO FILE... qm = {}  qu = {}  var1 = {}  var2 = {}".format(qm, qu, Delta1, Delta2)

				eff_den[qu] = cw_arrays[0]
				int_die[qu] += np.array((int_exx, int_ezz))
				cw_die[qu] += np.array((cw_exx1, cw_ezz1, cw_exx2, cw_ezz2, cw_arrays[5], cw_arrays[6], cw_arrays[7]**2, cw_arrays[8]**2))
				ellip_no[qu] += np.array((np.sqrt(cw_exx1), np.sqrt(cw_exx2), np.sqrt(cw_arrays[5]), cw_arrays[7]))


			with file('{}/INTDEN/{}_EFF_DEN.npy'.format(directory, file_name_den[r]), 'w') as outfile:
				np.save(outfile, eff_den)
			with file('{}/INTDIELEC/{}_DIE.npy'.format(directory, file_name_die[r]), 'w') as outfile:
				np.save(outfile, int_die)
			with file('{}/INTDIELEC/{}_CWDIE.npy'.format(directory, file_name_die[r]), 'w') as outfile:
				np.save(outfile, cw_die)
			with file('{}/ELLIP/{}_ELLIP_NO.npy'.format(directory, file_name_die[r]), 'w') as outfile:
				np.save(outfile, ellip_no)


	print "INTRINSIC SAMPLING METHOD {} {} {} {} {} COMPLETE\n".format(directory, model.upper(), qm, n0, phi)

	return av_auv1, av_auv2 
