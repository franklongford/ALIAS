"""
*************** INTRINSIC SAMPLING METHOD MODULE *******************

Performs intrinsic sampling analysis on a set of interfacial 
simulation configurations

********************************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 27/2/2018 by Frank Longford
"""

import numpy as np
import scipy as sp, scipy.constants as con

import utilities as ut

import os, sys, time, tables


def check_uv(u, v):
	"""
	check_uv(u, v)

	Returns weightings for frequencies u and v for anisotropic surfaces
	
	"""

	if abs(u) + abs(v) == 0: return 4.
	elif u * v == 0: return 2.
	else: return 1.


vcheck = np.vectorize(check_uv)


def wave_function(x, u, Lx):
	"""
	wave_function(x, u, Lx)

	Wave in Fouier sum 
	
	"""

	if u >= 0: return np.cos(2 * np.pi * u * x / Lx)
	else: return np.sin(2 * np.pi * abs(u) * x / Lx)


def d_wave_function(x, u, Lx):
	"""
	d_wave_function(x, u, Lx)

	Derivative of wave in Fouier sum wrt x
	
	"""

	if u >= 0: return - 2 * np.pi * u  / Lx * np.sin(2 * np.pi * u * x  / Lx)
	else: return 2 * np.pi * abs(u) / Lx * np.cos(2 * np.pi * abs(u) * x / Lx)


def dd_wave_function(x, u, Lx):
	"""
	dd_wave_function(x, u, Lx)

	Second derivative of wave in Fouier sum wrt x
	
	"""

	return - 4 * np.pi**2 * u**2 / Lx**2 * wave_function(x, u, Lx)


def update_A_b(xmol, ymol, zmol, dim, qm, n_waves, new_piv1, new_piv2):
	"""
	update_A_b(xmol, ymol, zmol, dim, qm, n_waves, new_piv1, new_piv2)
	
	Update A matrix and b vector for new pivot selection

	Paramters
	---------

	xmol:  float, array_like; shape=(nmol)
		Molecular coordinates in x dimension
	ymol:  float, array_like; shape=(nmol)
		Molecular coordinates in y dimension
	zmol:  float, array_like; shape=(nmol)
		Molecular coordinates in z dimension
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	n_waves:  int
		Number of coefficients / waves in surface
	new_piv1:  int, array_like
		Indices of new pivot molecules for surface 1
	new_piv2:  int, array_like
		Indices of new pivot molecules for surface 2

	Returns
	-------

	A:  float, array_like; shape=(2, n_waves**2, n_waves**2)
		Matrix containing wave product weightings f(x, u1, Lx).f(y, v1, Ly).f(x, u2, Lx).f(y, v2, Ly) 
		for each coefficient in the linear algebra equation Ax = b for both surfaces 
	b:  float, array_like; shape=(2, n_waves**2)
		Vector containing solutions z.f(x, u, Lx).f(y, v, Ly) to the linear algebra equation Ax = b 
		for both surfaces

	"""

	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

	A = np.zeros((2, n_waves**2, n_waves**2))
	b = np.zeros((2, n_waves**2))

	fuv1 = np.zeros((n_waves**2, len(new_piv1)))
	fuv2 = np.zeros((n_waves**2, len(new_piv2)))

	for j in xrange(n_waves**2):
		fuv1[j] = wave_function(xmol[new_piv1], u_array[j], dim[0]) * wave_function(ymol[new_piv1], v_array[j], dim[1])
		b[0][j] += np.sum(zmol[new_piv1] * fuv1[j])
		fuv2[j] = wave_function(xmol[new_piv2], u_array[j], dim[0]) * wave_function(ymol[new_piv2], v_array[j], dim[1])
		b[1][j] += np.sum(zmol[new_piv2] * fuv2[j])

	A[0] += np.dot(fuv1, fuv1.T)
	A[1] += np.dot(fuv2, fuv2.T)

	return A, b


def LU_decomposition(A, b):
	"""
	LU_decomposition(A, b)

	Perform lower-upper decomposition to solve equation Ax = b using scipy linalg lover-upper solver

	Parameters
	----------

	A:  float, array_like; shape=(2, n_waves**2, n_waves**2)
		Matrix containing wave product weightings f(x, u1, Lx).f(y, v1, Ly).f(x, u2, Lx).f(y, v2, Ly) 
		for each coefficient in the linear algebra equation Ax = b for both surfaces 
	b:  float, array_like; shape=(2, n_waves**2)
		Vector containing solutions z.f(x, u, Lx).f(y, v, Ly) to the linear algebra equation Ax = b 
		for both surfaces

	Returns
	-------

	coeff:	array_like (float); shape=(n_waves**2)
		Optimised surface coefficients	

	"""
	lu, piv  = sp.linalg.lu_factor(A)
	coeff = sp.linalg.lu_solve((lu, piv), b)

	return coeff


def make_zeta_list(xmol, ymol, dim, mol_list, coeff, qm, qu):
	"""
	zeta_list(xmol, ymol, dim, mol_list, coeff, qm)

	Calculate dz (zeta) between molecular sites and intrinsic surface for resolution qu"

	Parameters
	----------
	
	xmol:  float, array_like; shape=(nmol)
		Molecular coordinates in x dimension
	ymol:  float, array_like; shape=(nmol)
		Molecular coordinates in y dimension
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	mol_list:  int, array_like; shape=(n0)
		Indices of molcules available to be slected as pivots
	coeff:	array_like (float); shape=(n_waves**2)
		Optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	
	Returns
	-------

	zeta_list:  float, array_like; shape=(n0)
		Array of dz (zeta) between molecular sites and intrinsic surface
	
	"""

	zeta_list = xi(xmol[mol_list], ymol[mol_list], coeff, qm, qu, dim)
   
	return zeta_list


def pivot_selection(zmol, mol_sigma, n0, mol_list, zeta_list, piv_n, tau):
	"""
	pivot_selection(zmol, mol_sigma, n0, mol_list, zeta_list, piv_n, tau)

	Search through zeta_list for values within tau threshold and add to pivot list

	Parameters
	----------
	
	zmol:  float, array_like; shape=(nmol)
		Molecular coordinates in z dimension
	mol_sigma:  float
		Radius of spherical molecular interaction sphere
	n0:  int
		Maximum number of molecular pivots in intrinsic surface
	mol_list:  int, array_like; shape=(n0)
		Indices of molcules available to be selected as pivots
	zeta_list:  float, array_like; shape=(n0)
		Array of dz (zeta) between molecular sites and intrinsic surface
	piv_n:  int, array_like; shape=(n0)
		Molecular pivot indices
	tau:  float
		Threshold length along z axis either side of existing intrinsic surface for selection of new pivot points

	Returns
	-------

	mol_list:  int, array_like; shape=(n0)
		Updated indices of molcules available to be selected as pivots
	new_piv:  int, array_like
		Indices of new pivot molecules just selected
	piv_n:  int, array_like; shape=(n0)
		Updated molecular pivot indices

	"""

	"Find new pivots based on zeta <= tau"
	zeta = np.abs(zmol[mol_list] - zeta_list)	
	new_piv = mol_list[zeta <= tau]
	dz_new_piv = zeta[zeta <= tau]

	"Order pivots by zeta (shortest to longest)"
	ut.bubble_sort(new_piv, dz_new_piv)

	"Add new pivots to pivoy list and check whether max n0 pivots are selected"
	piv_n = np.concatenate((piv_n, new_piv))

	if piv_n.shape[0] > n0:
		new_piv = new_piv[:-piv_n.shape[0]+n0]
		piv_n = piv_n[:n0] 

	far_tau = 6.0 * tau
	
	"Remove pivots far from molecular search list"
	far_piv = mol_list[zeta > far_tau]
	if len(new_piv) > 0: mol_list = ut.numpy_remove(mol_list, np.concatenate((new_piv, far_piv)))
	
	assert np.sum(np.isin(new_piv, mol_list)) == 0

	return mol_list, new_piv, piv_n


def intrinsic_area(coeff, qm, qu, dim):
	"""
	intrinsic_area(coeff, qm, qu, dim)

	Calculate the intrinsic surface area from coefficients at resolution qu

	Parameters
	----------

	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	int_A:  float
		Relative size of intrinsic surface area, compared to cell cross section XY
	"""

	n_waves = 2 * qm +1
	
	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm
	wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
	indices = np.argwhere(wave_check).flatten()

	int_A = np.pi**2  * vcheck(u_array[indices], v_array[indices]) * (u_array[indices]**2 / dim[0]**2 + v_array[indices]**2 / dim[1]**2) * coeff[indices]**2
	int_A = 1 + 0.5 * np.sum(int_A)

	return int_A


def build_surface(xmol, ymol, zmol, dim, mol_sigma, qm, n0, phi, tau, max_r, ncube=3, vlim=3):
					
	"""
	build_surface(xmol, ymol, zmol, dim, nmol, mol_sigma, qm, n0, phi, tau, max_r, ncube=3, vlim=3)

	Create coefficients for Fourier sum representing intrinsic surface.

	Parameters
	----------

	xmol:  float, array_like; shape=(nmol)
		Molecular coordinates in x dimension
	ymol:  float, array_like; shape=(nmol)
		Molecular coordinates in y dimension
	zmol:  float, array_like; shape=(nmol)
		Molecular coordinates in z dimension
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	mol_sigma:  float
		Radius of spherical molecular interaction sphere
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	n0:  int
		Maximum number of molecular pivot in intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	ncube:	int (optional)
		Grid size for initial pivot molecule selection
	vlim:  int (optional)
		Minimum number of molecular meighbours within radius max_r required for molecular NOT to be considered in vapour region
	tau:  float
		Threshold length along z axis either side of existing intrinsic surface for selection of new pivot points
	max_r:  float
		Maximum radius for selection of vapour phase molecules

	Returns
	-------

	coeff:	array_like (float); shape=(2, n_waves**2)
		Optimised surface coefficients
	pivot:  array_like (int); shape=(2, n0)
		Indicies of pivot molecules in molecular position arrays	

	""" 

	nmol = len(xmol)
	tau1 = tau
	tau2 = tau
	mol_list = np.arange(nmol)
	piv_n1 = np.arange(ncube**2)
	piv_n2 = np.arange(ncube**2)
	piv_z1 = np.zeros(ncube**2)
        piv_z2 = np.zeros(ncube**2)
	vapour_list = []
	new_piv1 = []
	new_piv2 = []

	start = time.time()
	
	n_waves = 2*qm+1

	"Form the diagonal xi^2 terms"
	u = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

	diag = vcheck(u, v) * (u**2 * dim[1] / dim[0] + v**2 * dim[0] / dim[1])
	diag = 4 * np.pi**2 * phi * np.diagflat(diag)

	"Create empty A matrix and b vector for linear algebra equation Ax = b"
	A = np.zeros((2, n_waves**2, n_waves**2))
	b = np.zeros((2, n_waves**2))
	coeff = np.zeros((2, n_waves**2))

	if vlim == 0:
		piv_n1 = mol_list[zmol[mol_list] < 0]
		piv_n2 = mol_list[zmol[mol_list] >= 0]

		assert len(piv_n1) == n0
		assert len(piv_n2) == n0

		start1 = time.time()

		print "{:^55s} | {:^21s} | {:^21s}".format('TIMINGS (s)', 'PIVOTS', 'INT AREA')
		print ' {:20s}  {:20s}  {:10s} | {:10s} {:10s} | {:10s} {:10s}'.format('Matrix Formation', 'LU Decomposition', 'TOTAL', 'n_piv1', 'n_piv2','surf1', 'surf2')
		print "_" * 105

		"Update A matrix and b vector"
		temp_A, temp_b = update_A_b(xmol, ymol, zmol, dim, qm, n_waves, piv_n1, piv_n2)

		A += temp_A
		b += temp_b

		end1 = time.time()

		"Perform LU decomosition to solve Ax = b"
		coeff[0] = LU_decomposition(A[0] + diag, b[0])
		coeff[1] = LU_decomposition(A[1] + diag, b[1])

		end = time.time()

		"Calculate surface areas excess"
		area1 = intrinsic_area(coeff[0], qm, qm, dim)
		area2 = intrinsic_area(coeff[1], qm, qm, dim)

		print ' {:20.3f}  {:20.3f}  {:10.3f} | {:10d} {:10d} | {:10.3f} {:10.3f}'.format(end1 - start1, end - end1, end - start, len(piv_n1), len(piv_n2), area1, area2)			

	else:
		"Remove molecules from vapour phase ans assign an initial grid of pivots furthest away from centre of mass"
		print 'Lx = {:5.3f}   Ly = {:5.3f}   qm = {:5d}\nphi = {}   n_piv = {:5d}   vlim = {:5d}   max_r = {:5.3f}'.format(dim[0], dim[1], qm, phi, n0, vlim, max_r) 
		print 'Removing vapour molecules'

		dxyz = np.reshape(np.tile(np.stack((xmol, ymol, zmol)), (1, nmol)), (3, nmol, nmol))
		dxyz = np.transpose(dxyz, axes=(0, 2, 1)) - dxyz
		for i, l in enumerate(dim): dxyz[i] -= l * np.array(2 * dxyz[i] / l, dtype=int)
		dr2 = np.sum(dxyz**2, axis=0)

		vapour_list = np.where(np.count_nonzero(dr2 < max_r**2, axis=1) < vlim)

		del dxyz, dr2

		mol_list = ut.numpy_remove(mol_list, vapour_list)

		print 'Selecting initial {} pivots'.format(ncube**2)

		index_x = np.array(xmol * ncube / dim[0], dtype=int) % ncube
		index_y = np.array(ymol * ncube / dim[1], dtype=int) % ncube
		
		for n in mol_list:
			if zmol[n] < piv_z1[ncube*index_x[n] + index_y[n]]: 
				piv_n1[ncube*index_x[n] + index_y[n]] = n
				piv_z1[ncube*index_x[n] + index_y[n]] = zmol[n]
			elif zmol[n] > piv_z2[ncube*index_x[n] + index_y[n]]: 
				piv_n2[ncube*index_x[n] + index_y[n]] = n
				piv_z2[ncube*index_x[n] + index_y[n]] = zmol[n]

		"Update molecular and pivot lists"
		mol_list = ut.numpy_remove(mol_list, piv_n1)
		mol_list = ut.numpy_remove(mol_list, piv_n2)

		new_piv1 = piv_n1
		new_piv2 = piv_n2

		assert np.sum(np.isin(vapour_list, mol_list)) == 0
		assert np.sum(np.isin(piv_n1, mol_list)) == 0
		assert np.sum(np.isin(piv_n2, mol_list)) == 0

		print 'Initial {} pivots selected: {:10.3f} s'.format(ncube**2, time.time() - start)

		"Split molecular position lists into two volumes for each surface"
		mol_list1 = mol_list[zmol[mol_list] < 0]
		mol_list2 = mol_list[zmol[mol_list] >= 0]

		assert piv_n1 not in mol_list1
		assert piv_n2 not in mol_list2

		print "{:^77s} | {:^43s} | {:^21s} | {:^21s}".format('TIMINGS (s)', 'PIVOTS', 'TAU', 'INT AREA')
		print ' {:20s}  {:20s}  {:20s}  {:10s} | {:10s} {:10s} {:10s} {:10s} | {:10s} {:10s} | {:10s} {:10s}'.format('Matrix Formation', 'LU Decomposition', 'Pivot selection', 'TOTAL', 'n_piv1', '(new)', 'n_piv2', '(new)', 'surf1', 'surf2', 'surf1', 'surf2')
		print "_" * 170

		building_surface = True
		build_surf1 = True
		build_surf2 = True

		while building_surface:

			start1 = time.time()

			"Update A matrix and b vector"
			temp_A, temp_b = update_A_b(xmol, ymol, zmol, dim, qm, n_waves, new_piv1, new_piv2)

			A += temp_A
			b += temp_b

			end1 = time.time()

			"Perform LU decomosition to solve Ax = b"
			if len(new_piv1) != 0: coeff[0] = LU_decomposition(A[0] + diag, b[0])
			if len(new_piv2) != 0: coeff[1] = LU_decomposition(A[1] + diag, b[1])

			end2 = time.time()

			"Check whether more pivots are needed"
			if len(piv_n1) == n0: 
				build_surf1 = False
				new_piv1 = []
			if len(piv_n2) == n0: 
				build_surf2 = False
				new_piv2 = []

			if build_surf1 or build_surf2:
			        finding_pivots = True
			        piv_search1 = True
			        piv_search2 = True
			else:
			        finding_pivots = False
			        building_surface = False
			        print "ENDING SEARCH"

			"Calculate distance between molecular z positions and intrinsic surface"
			if build_surf1: zeta_list1 = make_zeta_list(xmol, ymol, dim, mol_list1, coeff[0], qm, qm)
			if build_surf2: zeta_list2 = make_zeta_list(xmol, ymol, dim, mol_list2, coeff[1], qm, qm)

			"Search for more molecular pivot sites"
			while finding_pivots:

				"Perform pivot selectrion"
				if piv_search1 and build_surf1: mol_list1, new_piv1, piv_n1 = pivot_selection(zmol, mol_sigma, n0, mol_list1, zeta_list1, piv_n1, tau1)
				if piv_search2 and build_surf2: mol_list2, new_piv2, piv_n2 = pivot_selection(zmol, mol_sigma, n0, mol_list2, zeta_list2, piv_n2, tau2)

				"Check whether threshold distance tau needs to be increased"
			        if len(new_piv1) == 0 and len(piv_n1) < n0: tau1 += 0.1 * tau 
				else: piv_search1 = False

			        if len(new_piv2) == 0 and len(piv_n2) < n0: tau2 += 0.1 * tau 
				else: piv_search2 = False

				if piv_search1 or piv_search2: finding_pivots = True
			        else: finding_pivots = False

			end = time.time()

			"Calculate surface areas excess"
			area1 = intrinsic_area(coeff[0], qm, qm, dim)
			area2 = intrinsic_area(coeff[1], qm, qm, dim)

			print ' {:20.3f}  {:20.3f}  {:20.3f}  {:10.3f} | {:10d} {:10d} {:10d} {:10d} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f}'.format(end1 - start1, end2 - end1, end - end2, end - start1, len(piv_n1), len(new_piv1), len(piv_n2), len(new_piv2), tau1, tau2, area1, area2)			

	print '\nTOTAL time: {:7.2f} s \n'.format(end - start)

	pivot = np.array((piv_n1, piv_n2), dtype=int)

	#ut.view_surface(coeff, pivot, qm, qm, xmol, ymol, zmol, 30, dim)

	return coeff, pivot


def surface_reconstruction(coeff, pivot, xmol, ymol, zmol, dim, qm, n0, phi, psi, precision=1E-3, max_step=20):
	"""
	surface_reconstruction( xmol, ymol, zmol, dim, qm, n0, phi, psi)

	Reconstruct surface coefficients in Fourier sum representing intrinsic surface to yield expected variance of mean curvature.

	Parameters
	----------

	coeff:	array_like (float); shape=(2, n_waves**2)
		Optimised surface coefficients
	pivot:  array_like (int); shape=(2, n0)
		Indicies of pivot molecules in molecular position arrays
	xmol:  float, array_like; shape=(nmol)
		Molecular coordinates in x dimension
	ymol:  float, array_like; shape=(nmol)
		Molecular coordinates in y dimension
	zmol:  float, array_like; shape=(nmol)
		Molecular coordinates in z dimension
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	n0:  int
		Maximum number of molecular pivots in intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	psi:  float
		Initial value of weighting factor for surface reconstruction routine

	Returns
	-------

	coeff_recon:  array_like (float); shape=(2, n_waves**2)
		Reconstructed surface coefficients
	""" 

	n_waves = 2*qm+1

	print "PERFORMING SURFACE RESTRUCTURING"
	print 'Lx = {:5.3f}   Ly = {:5.3f}   qm = {:5d}\nphi = {}  psi = {}  n_piv = {:5d}  precision = {}'.format(dim[0], dim[1], qm, phi, psi, n0, precision) 
	print 'Setting up wave product and coefficient matricies'

	start = time.time()	

	"Form the diagonal xi^2 terms and b vector solutions"
	A = np.zeros((2, n_waves**2, n_waves**2))
	fuv = np.zeros((2, n_waves**2, n0))
	ffuv = np.zeros((2, n_waves**2, n_waves**2))
	b = np.zeros((2, n_waves**2))

	for i in xrange(2):
		for j in xrange(n_waves**2):
		        fuv[i][j] = wave_function(xmol[pivot[i]], int(j/n_waves)-qm, dim[0]) * wave_function(ymol[pivot[i]], int(j%n_waves)-qm, dim[1])
		        b[i][j] += np.sum(zmol[pivot[i]] * fuv[i][j])

		ffuv[i] = np.dot(fuv[i], fuv[i].T)

	"Create arrays of wave frequency indicies u and v"
	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

	uv_check = vcheck(u_array, v_array)

	"Make diagonal terms of A matrix"
	diag = uv_check * (phi * (u_array**2 * dim[1] / dim[0] + v_array**2 * dim[0] / dim[1]))
	diag = 4 * np.pi**2 * np.diagflat(diag)

	"Create matrix of wave frequency indicies (u,v)**2"
	u_matrix = np.tile(u_array, (n_waves**2, 1))
	v_matrix = np.tile(v_array, (n_waves**2, 1))

	"Make curvature diagonal terms of A matrix"
	curve_diag = 16 * np.pi**4 * (u_matrix**2 * u_matrix.T**2 / dim[0]**4 + v_matrix**2 * v_matrix.T**2 / dim[1]**4 +
				     (u_matrix**2 * v_matrix.T**2 + u_matrix.T**2 * v_matrix**2) / (dim[0]**2 * dim[1]**2))

	end_setup1 = time.time()

	print "{:^74s} | {:^21s} | {:^43s} | {:^21s}".format('TIMINGS (s)', 'PSI', 'VAR(H)', 'INT AREA' )
	print ' {:20s} {:20s} {:20s} {:10s} | {:10s} {:10s} | {:10s} {:10s} {:10s} {:10s} | {:10s} {:10s}'.format('Matrix Formation', 'LU Decomposition', 
				'Var Estimation', 'TOTAL', 'surf1', 'surf2', 'surf1', 'piv1', 'surf2', 'piv2','surf1', 'surf2')
	print "_" * 168

	H_var_coeff = np.zeros((2, 2))
	H_var_piv = np.zeros((2, 2))
	H_var_func = np.zeros((2, 2))
	H_var_grad = np.zeros((2, 2))
	area = np.zeros((2))

	H_var = 4 * np.pi**4 * uv_check * (u_array**4 / dim[0]**4 + v_array**4 / dim[1]**4 + 2 * u_array**2 * v_array**2 / (dim[0]**2 * dim[1]**2))
	for i in xrange(2): 
		"Calculate variance of curvature across entire surface from coefficients"
		H_var_coeff[0][i] = np.sum(H_var * coeff[i]**2)
		"Calculate variance of curvature at pivot sites only"
		coeff_matrix = np.tile(coeff[i], (n_waves**2, 1))
		H_var_piv[0][i] = np.sum(coeff_matrix * coeff_matrix.T * ffuv[i] * curve_diag / n0)
		"Calculate intrinsic surface area"
		area[i] = intrinsic_area(coeff[i], qm, qm, dim)
		"Calculate optimisation function (diff between coeff and pivot variance)"
		H_var_func[0][i] = abs(H_var_coeff[0][i] - H_var_piv[0][i])
		"Calculate gradient of optimistation function wrt psi"
		H_var_grad[0][i] = 1

	end_setup2 = time.time()

	print ' {:20.3f} {:20.3f} {:20.3f} {:10.3f} | {:10.6f} {:10.6f} | {:10.4f} {:10.4f} {:10.4f} {:10.4f} | {:10.3f} {:10.3f}'.format(end_setup1-start, 
			0, end_setup2-end_setup1, end_setup2-start, 0, 0, H_var_coeff[0][0], H_var_piv[0][0], H_var_coeff[0][1], H_var_piv[0][1], area[0], area[1])

	reconstructing = True
	recon_array = [True, True]
	psi_array = np.array([(0, 0), (psi, psi)])
	coeff_recon = np.zeros(coeff.shape)
	step = np.zeros(2)
	weight = np.ones(2) * 0.9
	
	"Amend psi weighting coefficient until H_var == H_piv_var"
	while reconstructing:

		start1 = time.time()
        
		"Update A matrix and b vector" 
		for i in xrange(2): A[i] = ffuv[i] * (1. + curve_diag * psi_array[1][i] / n0)

		end1 = time.time()

		"Update coeffs by performing LU decomosition to solve Ax = b"
		for i, recon in enumerate(recon_array):
			if recon: coeff_recon[i] = LU_decomposition(A[i] + diag, b[i])

		end2 = time.time()

		for i, recon in enumerate(recon_array):
			"Recalculate variance of curvature across entire surface from coefficients"
			H_var_coeff[1][i] = np.sum(H_var * coeff_recon[i]**2)
			if recon:
				"Recalculate variance of curvature at pivot sites only"
				coeff_matrix_recon = np.tile(coeff_recon[i], (n_waves**2, 1))
				H_var_piv[1][i] = np.sum(coeff_matrix_recon * coeff_matrix_recon.T * ffuv[i] * curve_diag / n0)
				"Recalculate intrinsic surface area"
				area[i] = intrinsic_area(coeff_recon[i], qm, qm, dim)
				"Recalculate optimisation function (diff between coeff and pivot variance)"
				H_var_func[1][i] = abs(H_var_coeff[1][i] - H_var_piv[1][i])
				"Recalculate gradient of optimistation function wrt psi"
				H_var_grad[1][i] = (H_var_func[1][i] - H_var_func[0][i]) / (psi_array[1][i] - psi_array[0][i])

		end3 = time.time()

		print ' {:20.3f} {:20.3f} {:20.3f} {:10.3f} | {:10.6f} {:10.6f} | {:10.4f} {:10.4f} {:10.4f} {:10.4f} | {:10.3f} {:10.3f}'.format(end1 - start1, 
				end2 - end1, end3 - end2, end3 - start1, psi_array[1][0], psi_array[1][1],  H_var_coeff[1][0], H_var_piv[1][0], H_var_coeff[1][1], H_var_piv[1][1], area[0], area[1])


		for i, recon in enumerate(recon_array):
			if abs(H_var_func[1][i]) <= precision: recon_array[i] = False
			else:
				step[i] += 1

				if step[i] >= max_step:
					"Reconstruction routine failed to find minimum. Restart using smaller psi"
					psi_array[0][i] = 0
					psi_array[1][i] = psi * weight[i]
					"Calculate original values of Curvature variances"
					H_var_coeff[0][i] = np.sum(H_var * coeff[i]**2)
					H_var_piv[0][i] = np.sum(coeff_matrix * coeff_matrix.T * ffuv[i] * curve_diag / n0)
					area[i] = intrinsic_area(coeff[i], qm, qm, dim)
					H_var_func[0][i] = abs(H_var_coeff[0][i] - H_var_piv[0][i])
					H_var_grad[0][i] = 1
					"Decrease psi weighting for next run"
					weight[i] *= 0.9
					"Reset number of steps"
					step[i] = 0
				else:
					gamma =  H_var_func[1][i] / H_var_grad[1][i]

					psi_array[0][i] = psi_array[1][i]
					psi_array[1][i] -= gamma

					H_var_coeff[0][i] = H_var_coeff[1][i]
					H_var_piv[0][i] = H_var_piv[1][i]
					H_var_func[0][i] = H_var_func[1][i]
					H_var_grad[0][i] = H_var_grad[1][i]

		if not np.any(recon_array): reconstructing = False

	end = time.time()

	print '\nTOTAL time: {:7.2f} s \n'.format(end - start)

	return coeff_recon


def wave_function_array(x, u_array, Lx):
	"""
	wave_function_array(x, u_array, Lx)

	Returns numpy array of all waves in Fouier sum 
	
	"""

	q = 2 * np.pi * np.abs(u_array) * x / Lx

	cos_indicies = np.argwhere(u_array >= 0)
	sin_indicies = np.argwhere(u_array < 0)
	f_array = np.zeros(u_array.shape)
	f_array[cos_indicies] += np.cos(q[cos_indicies])
	f_array[sin_indicies] += np.sin(q[sin_indicies])

	return f_array


def d_wave_function_array(x, u_array, Lx):
	"""
	d_wave_function_array(x, u_array, Lx)

	Returns numpy array of all derivatives of waves in Fouier sum 
	
	"""

	q = 2 * np.pi * np.abs(u_array) * x / Lx

	cos_indicies = np.argwhere(u_array >= 0)
	sin_indicies = np.argwhere(u_array < 0)
	f_array = np.zeros(u_array.shape)
	f_array[cos_indicies] -= np.sin(q[cos_indicies])
	f_array[sin_indicies] += np.cos(q[sin_indicies])
	f_array *= 2 * np.pi * np.abs(u_array) / Lx

	return f_array


def dd_wave_function_array(x, u_array, Lx):
	"""
	dd_wave_function_array(x, u_array, Lx)
	Returns numpy array of all second derivatives of waves in Fouier sum 
	
	"""
	return - 4 * np.pi**2 * u_array**2 / Lx**2 * wave_function_array(x, u_array, Lx)


def xi(x, y, coeff, qm, qu, dim):
	"""
	xi(x, y, coeff, qm, qu, dim)

	Function returning position of intrinsic surface at position (x,y)

	Parameters
	----------

	x:  float, array_like; shape=(nmol)
		Coordinate in x dimension
	y:  float, array_like; shape=(nmol)
		Coordinate in y dimension
	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	xi_z:  float, array_like; shape=(nmol)
		Positions of intrinsic surface in z dimension

	"""

	n_waves = 2 * qm + 1
	
	if np.isscalar(x):
		u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
		v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
		wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
		indices = np.argwhere(wave_check).flatten()

		fuv = wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
		xi_z = np.sum(fuv * coeff[indices])
	else:
		xi_z = np.zeros(x.shape)
		for u in xrange(-qu, qu+1):
			for v in xrange(-qu, qu+1):
				j = (2 * qm + 1) * (u + qm) + (v + qm)
				xi_z += wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]	
	return xi_z


def dxy_dxi(x, y, coeff, qm, qu, dim):
	"""
	dxy_dxi(x, y, qm, qu, coeff, dim)

	Function returning derivatives of intrinsic surface at position (x,y) wrt x and y

	Parameters
	----------

	x:  float
		Coordinate in x dimension
	y:  float
		Coordinate in y dimension
	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	dx_dxi:  float
		Derivative of intrinsic surface in x dimension
	dy_dxi:  float
		Derivative of intrinsic surface in y dimension
	
	"""

	n_waves = 2 * qm + 1

	if np.isscalar(x):
		u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
		v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
		wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
		indices = np.argwhere(wave_check).flatten()

		dx_dxi = d_wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
		dy_dxi = wave_function_array(x, u_array[indices], dim[0]) * d_wave_function_array(y, v_array[indices], dim[1])

		dx_dxi = np.sum(dx_dxi * coeff[indices])
		dy_dxi = np.sum(dy_dxi * coeff[indices])

	else:
		dx_dxi = np.zeros(x.shape)
		dy_dxi = np.zeros(x.shape)
		for u in xrange(-qu, qu+1):
			for v in xrange(-qu, qu+1):
				j = (2 * qm + 1) * (u + qm) + (v + qm)
				dx_dxi += d_wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
				dy_dxi += wave_function(x, u, dim[0]) * d_wave_function(y, v, dim[1]) * coeff[j]

	return dx_dxi, dy_dxi


def ddxy_ddxi(x, y, coeff, qm, qu, dim):
	"""
	ddxy_ddxi(x, y, coeff, qm, qu, dim)

	Function returning second derivatives of intrinsic surface at position (x,y) wrt x and y
	
	Parameters
	----------

	x:  float
		Coordinate in x dimension
	y:  float
		Coordinate in y dimension
	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	ddx_ddxi:  float
		Second derivative of intrinsic surface in x dimension
	ddy_ddxi:  float
		Second derivative of intrinsic surface in y dimension
	
	"""

	n_waves = 2 * qm + 1

	if np.isscalar(x):
		u_array = (np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm)
		v_array = (np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm)
		wave_check = (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)
		indices = np.argwhere(wave_check).flatten()

		ddx_ddxi = dd_wave_function_array(x, u_array[indices], dim[0]) * wave_function_array(y, v_array[indices], dim[1])
		ddy_ddxi = wave_function_array(x, u_array[indices], dim[0]) * dd_wave_function_array(y, v_array[indices], dim[1])

		ddx_ddxi = np.sum(ddx_ddxi * coeff[indices])
		ddy_ddxi = np.sum(ddy_ddxi * coeff[indices])

	else:
		ddx_ddxi = np.zeros(x.shape)
		ddy_ddxi = np.zeros(x.shape)
		for u in xrange(-qu, qu+1):
			for v in xrange(-qu, qu+1):
				j = (2 * qm + 1) * (u + qm) + (v + qm)
				ddx_ddxi += dd_wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
				ddy_ddxi += wave_function(x, u, dim[0]) * dd_wave_function(y, v, dim[1]) * coeff[j]

	return ddx_ddxi, ddy_ddxi


def optimise_ns(directory, file_name, nmol, nframe, qm, phi, dim, mol_sigma, start_ns, step_ns, AIC=False, nframe_ns = 20, ncube=3, vlim=3, tau=0.5, max_r=1.5, precision=0.0005, gamma=0.5):
	"""
	optimise_ns(directory, file_name, nmol, nframe, qm, phi, ncube, dim, mol_sigma, start_ns, step_ns, nframe_ns = 20, vlim=3)

	Routine to find optimised pivot density coefficient ns and pivot number n0 based on lowest pivot diffusion rate

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed.
	nmol:  int
		Number of molecules in simulation
	nframe: int
		Number of trajectory frames in simulation
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	mol_sigma:  float
		Radius of spherical molecular interaction sphere
	start_ns:  float
		Initial value for pivot density optimisation routine
	step_ns:  float
		Search step difference between each pivot density value ns
	nframe_ns:  (optional) int
		Number of trajectory frames to perform optimisation with

	Returns
	-------

	opt_ns: float
		Optimised surface pivot density parameter
	opt_n0: int
		Optimised number of pivot molecules

	"""

	pos_dir = directory + 'pos/'
	surf_dir = directory + 'surface/'

	if not os.path.exists(surf_dir): os.mkdir(surf_dir)

	mol_ex_1 = []
	mol_ex_2 = []
	NS = []
	derivative = []

	n_waves = 2 * qm + 1
	max_r *= mol_sigma
	tau *= mol_sigma
	
	xmol = ut.load_npy(pos_dir + file_name + '_{}_xmol'.format(nframe), frames=range(nframe_ns))
	ymol = ut.load_npy(pos_dir + file_name + '_{}_ymol'.format(nframe), frames=range(nframe_ns))
	zmol = ut.load_npy(pos_dir + file_name + '_{}_zmol'.format(nframe), frames=range(nframe_ns))
	COM = ut.load_npy(pos_dir + file_name + '_{}_com'.format(nframe), frames=range(nframe_ns))

	com_tile = np.moveaxis(np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
	zmol = zmol - com_tile

	if nframe < nframe_ns: nframe_ns = nframe
	ns = start_ns
	optimising = True

	print("Surface pivot density precision = {}".format(precision))

	while optimising:

		NS.append(ns)
		n0 = int(dim[0] * dim[1] * ns / mol_sigma**2)

		print("Density Coefficient = {}".format(ns))		
		print("Using pivot number = {}".format(n0))
		
		tot_piv_n1 = np.zeros((nframe_ns, n0), dtype=int)
		tot_piv_n2 = np.zeros((nframe_ns, n0), dtype=int)

		file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)

		if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_coeff)):
			ut.make_hdf5(surf_dir + file_name_coeff + '_coeff', (2, n_waves**2), tables.Float64Atom())
			ut.make_hdf5(surf_dir + file_name_coeff + '_pivot', (2, n0), tables.Int64Atom())

		for frame in xrange(nframe_ns):
			"Checking number of frames in coeff and pivot files"
			frame_check_coeff = (ut.shape_check_hdf5(surf_dir + file_name_coeff + '_coeff')[0] <= frame)
			frame_check_pivot = (ut.shape_check_hdf5(surf_dir + file_name_coeff + '_pivot')[0] <= frame)

			if frame_check_coeff: mode_coeff = 'a'
			else: mode_coeff = False

			if frame_check_pivot: mode_pivot = 'a'
			else: mode_pivot = False

			if not mode_coeff and not mode_pivot:
				pivot = ut.load_hdf5(surf_dir + file_name_coeff + '_pivot', frame)
			else:
				sys.stdout.write("Optimising Intrinsic Surface coefficients: frame {}\n".format(frame))
				sys.stdout.flush()

				coeff, pivot = build_surface(xmol[frame], ymol[frame], zmol[frame], dim, mol_sigma, qm, n0, phi, tau, max_r, ncube=ncube, vlim=vlim)
				ut.save_hdf5(surf_dir + file_name_coeff + '_coeff', coeff, frame, mode_coeff)
				ut.save_hdf5(surf_dir + file_name_coeff + '_pivot', pivot, frame, mode_pivot)

			tot_piv_n1[frame] += pivot[0]
			tot_piv_n2[frame] += pivot[1]

		ex_1, ex_2 = mol_exchange(tot_piv_n1, tot_piv_n2, nframe_ns, n0)

		mol_ex_1.append(ex_1)
		mol_ex_2.append(ex_2)

		if AIC: 
			av_mol_ex = 2 * n0 - np.log(np.array(mol_ex_1) + np.array(mol_ex_2))
			print("Average Pivot Diffusion AIC = {}".format(av_mol_ex[-1]))
		else: 
			av_mol_ex = (np.array(mol_ex_1) + np.array(mol_ex_2)) / 2.
			print("Average Pivot Diffusion Rate = {} mol / frame".format(av_mol_ex[-1]))

		if len(av_mol_ex) > 1:
			step_size = (av_mol_ex[-1] - av_mol_ex[-2])
			derivative.append(step_size / (NS[-1] - NS[-2]))
			#min_arg = np.argsort(abs(NS[-1] - np.array(NS)))
			#min_derivative = (av_mol_ex[min_arg[0]] - av_mol_ex[min_arg[1]]) / (NS[min_arg[0]] - NS[min_arg[1]])
			#derivative.append(min_derivative)
			
			check = abs(step_size) <= precision
			if check: optimising = False
			else:
				#if len(derivative) > 1: gamma = (NS[-1] - NS[-2]) / (derivative[-1] - derivative[-2])
				ns -= gamma * derivative[-1]
				print("Optimal pivot density not found.\nSurface density coefficient step size = |{}| > {}\n".format(step_size, precision))

		else: ns += step_ns
		
	opt_ns = NS[np.argmin(av_mol_ex)]
	opt_n0 = int(dim[0] * dim[1] * opt_ns / mol_sigma**2)

	print "Optimal pivot density found = {}\nSurface density coefficient step size = |{}| < {}\n".format(opt_ns, step_size, precision)
	print "Optimal number of pivots = {}".format(opt_n0)

	for ns in NS:
		if ns != opt_ns:
			n0 = int(dim[0] * dim[1] * ns / mol_sigma**2)
			file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
			os.remove(surf_dir + file_name_coeff + '_coeff.hdf5')
			os.remove(surf_dir + file_name_coeff + '_pivot.hdf5')

	return opt_ns, opt_n0


def mol_exchange(piv_1, piv_2, nframe, n0):
	"""
	mol_exchange(piv_1, piv_2, nframe, n0)

	Calculates average diffusion rate of surface pivot molecules between frames

	Parameters
	----------

	piv_1:  float, array_like; shape=(nframe, n0)
		Molecular pivot indicies of upper surface at each frame
	piv_2:  float, array_like; shape=(nframe, n0)
		Molecular pivot indicies of lower surface at each frame
	nframe:  int
		Number of frames to sample over
	n0:  int
		Number of pivot molecules in each surface

	Returns
	-------

	diff_rate1: float
		Diffusion rate of pivot molecules in mol frame^-1 of upper surface
	diff_rate2: float
		Diffusion rate of pivot molecules in mol frame^-1 of lower surface

	"""
	n_1 = 0
	n_2 = 0

	for frame in xrange(nframe-1):

		n_1 += len(set(piv_1[frame]) - set(piv_1[frame+1]))
		n_2 += len(set(piv_2[frame]) - set(piv_2[frame+1]))

	diff_rate1 = n_1 / (n0 * float(nframe-1))
	diff_rate2 = n_2 / (n0 * float(nframe-1))

	return diff_rate1, diff_rate2



def create_intrinsic_surfaces(directory, file_name, dim, qm, n0, phi, mol_sigma, nframe, recon=True, ncube=3, vlim=3, tau=0.5, max_r=1.5, ow_coeff=False, ow_recon=False):
	"""
	create_intrinsic_surfaces(directory, file_name, dim, qm, n0, phi, mol_sigma, nframe, recon=True, ow_coeff=False, ow_recon=False)

	Routine to find optimised pivot density coefficient ns and pivot number n0 based on lowest pivot diffusion rate

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed.
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	n0:  int
		Maximum number of molecular pivots in intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	mol_sigma:  float
		Radius of spherical molecular interaction sphere
	nframe:  int
		Number of frames in simulation trajectory
	recon:  bool (optional)
		Whether to perform surface reconstruction routine (default=True)
	ow_coeff:  bool (optional)
		Whether to overwrite surface coefficients (default=False)
	ow_recon:  bool (optional)
		Whether to overwrite reconstructed surface coefficients (default=False)

	"""

	print"\n--- Running Intrinsic Surface Routine ---\n"

	surf_dir = directory + 'surface/'
	pos_dir = directory + 'pos/'

	if not os.path.exists(surf_dir): os.mkdir(surf_dir)

	file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	n_waves = 2 * qm + 1
	max_r *= mol_sigma
	tau *= mol_sigma

	"Make coefficient and pivot files"
	if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_coeff)):
		ut.make_hdf5(surf_dir + file_name_coeff + '_coeff', (2, n_waves**2), tables.Float64Atom())
		ut.make_hdf5(surf_dir + file_name_coeff + '_pivot', (2, n0), tables.Int64Atom())
		file_check = False
	elif not ow_coeff:
		"Checking number of frames in current coefficient files"
		try:
			file_check = (ut.shape_check_hdf5(surf_dir + file_name_coeff + '_coeff') == (nframe, 2, n_waves**2))
			file_check *= (ut.shape_check_hdf5(surf_dir + file_name_coeff + '_pivot') == (nframe, 2, n0))
		except: file_check = False
	else: file_check = False

	if recon:
		psi = phi * dim[0] * dim[1]
		"Make recon coefficient file"
		if not os.path.exists('{}/surface/{}_R_coeff.hdf5'.format(directory, file_name_coeff)):
			ut.make_hdf5(surf_dir + file_name_coeff + '_R_coeff', (2, n_waves**2), tables.Float64Atom())
			file_check = False
		elif not ow_recon:
			"Checking number of frames in current recon coefficient files"
			try: file_check = (ut.shape_check_hdf5(surf_dir + file_name_coeff + '_R_coeff') == (nframe, 2, n_waves**2))
			except: file_check = False
		else: file_check = False

	if not file_check:
		print "IMPORTING GLOBAL POSITION DISTRIBUTIONS\n"
		xmol = ut.load_npy(pos_dir + file_name + '_{}_xmol'.format(nframe))
		ymol = ut.load_npy(pos_dir + file_name + '_{}_ymol'.format(nframe))
		zmol = ut.load_npy(pos_dir + file_name + '_{}_zmol'.format(nframe))
		COM = ut.load_npy(pos_dir + file_name + '_{}_com'.format(nframe))
		nmol = xmol.shape[1]
		com_tile = np.moveaxis(np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
		zmol = zmol - com_tile

		for frame in xrange(nframe):

			"Checking number of frames in coeff and pivot files"
			frame_check_coeff = (ut.shape_check_hdf5(surf_dir + file_name_coeff + '_coeff')[0] <= frame)
			frame_check_pivot = (ut.shape_check_hdf5(surf_dir + file_name_coeff + '_pivot')[0] <= frame)

			if frame_check_coeff: mode_coeff = 'a'
			elif ow_coeff: mode_coeff = 'r+'
			else: mode_coeff = False

			if frame_check_pivot: mode_pivot = 'a'
			elif ow_coeff: mode_pivot = 'r+'
			else: mode_pivot = False

			if not mode_coeff and not mode_pivot: pass
			else:
				sys.stdout.write("Optimising Intrinsic Surface coefficients: frame {}\n".format(frame))
				sys.stdout.flush()
                        
				coeff, pivot = build_surface(xmol[frame], ymol[frame], zmol[frame], dim, mol_sigma, qm, n0, phi, tau, max_r, ncube=ncube, vlim=vlim)

				#ut.view_surface(coeff[0], qm, qm, xmol[frame][pivot[0]], ymol[frame][pivot[0]], zmol[frame][pivot[0]], 50, dim)
				#ut.view_surface(coeff[1], qm, qm, xmol[frame][pivot[1]], ymol[frame][pivot[1]], zmol[frame][pivot[1]], 50, dim)

				ut.save_hdf5(surf_dir + file_name_coeff + '_coeff', coeff, frame, mode_coeff)
				ut.save_hdf5(surf_dir + file_name_coeff + '_pivot', pivot, frame, mode_pivot)

			if recon:
				frame_check_recon = (ut.shape_check_hdf5(surf_dir + file_name_coeff + '_R_coeff')[0] <= frame)

				if frame_check_recon: mode_recon = 'a'
				elif ow_coeff or ow_recon: mode_recon = 'r+'
				else: mode_recon = False

				if not mode_recon: pass
				else:
					sys.stdout.write("Reconstructing Intrinsic Surface coefficients: frame {}\r".format(frame))
					sys.stdout.flush()

					coeff = ut.load_hdf5(surf_dir + file_name_coeff + '_coeff', frame)
					pivot = ut.load_hdf5(surf_dir + file_name_coeff + '_pivot', frame)
					coeff_R = surface_reconstruction(coeff, pivot, xmol[frame], ymol[frame], zmol[frame], dim, qm, n0, phi, psi)
					ut.save_hdf5(surf_dir + file_name_coeff + '_R_coeff', coeff_R, frame, mode_coeff)


def make_pos_dxdy(directory, file_name_pos, xmol, ymol, coeff, nmol, dim, qm):
	"""
	make_pos_dxdy(directory, file_name_pos, xmol, ymol, coeff, nmol, dim, qm)

	Calculate distances and derivatives at each molecular position with respect to intrinsic surface

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name_pos:  str
		File name to save position and derivatives to.
	xmol:  float, array_like; shape=(nmol)
		Molecular coordinates in x dimension
	ymol:  float, array_like; shape=(nmol)
		Molecular coordinates in y dimension
	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	nmol:  int
		Number of molecules in simulation
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface

	Returns
	-------

	int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
		Molecular distances from intrinsic surface
	int_dxdy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
		First derivatives of intrinsic surface wrt x and y at xmol, ymol
	int_ddxddy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
		Second derivatives of intrinsic surface wrt x and y at xmol, ymol 

	"""
	
	int_z_mol = np.zeros((qm+1, 2, nmol))
	int_dxdy_mol = np.zeros((qm+1, 4, nmol)) 
	int_ddxddy_mol = np.zeros((qm+1, 4, nmol))

	temp_int_z_mol = np.zeros((2, nmol))
	temp_dxdy_mol = np.zeros((4, nmol)) 
	temp_ddxddy_mol = np.zeros((4, nmol))
	
	for qu in xrange(qm+1):

		if qu == 0:
			j = (2 * qm + 1) * qm + qm
			f_x = wave_function(xmol, 0, dim[0])
			f_y = wave_function(ymol, 0, dim[1])

			temp_int_z_mol[0] += f_x * f_y * coeff[0][j]
			temp_int_z_mol[1] += f_x * f_y * coeff[1][j]

		else:
			for u in [-qu, qu]:
				for v in xrange(-qu, qu+1):
					j = (2 * qm + 1) * (u + qm) + (v + qm)

					f_x = wave_function(xmol, u, dim[0])
					f_y = wave_function(ymol, v, dim[1])
					df_dx = d_wave_function(xmol, u, dim[0])
					df_dy = d_wave_function(ymol, v, dim[1])
					ddf_ddx = dd_wave_function(xmol, u, dim[0])
					ddf_ddy = dd_wave_function(ymol, v, dim[1])

					temp_int_z_mol[0] += f_x * f_y * coeff[0][j]
					temp_int_z_mol[1] += f_x * f_y * coeff[1][j]
					temp_dxdy_mol[0] += df_dx * f_y * coeff[0][j]
					temp_dxdy_mol[1] += f_x * df_dy * coeff[0][j]
					temp_dxdy_mol[2] += df_dx * f_y * coeff[1][j]
					temp_dxdy_mol[3] += f_x * df_dy * coeff[1][j]
					temp_ddxddy_mol[0] += ddf_ddx * f_y * coeff[0][j]
					temp_ddxddy_mol[1] += f_x * ddf_ddy * coeff[0][j]
					temp_ddxddy_mol[2] += ddf_ddx * f_y * coeff[1][j]
					temp_ddxddy_mol[3] += f_x * ddf_ddy * coeff[1][j]

			for u in xrange(-qu+1, qu):
				for v in [-qu, qu]:
					j = (2 * qm + 1) * (u + qm) + (v + qm)

					f_x = wave_function(xmol, u, dim[0])
					f_y = wave_function(ymol, v, dim[1])
					df_dx = d_wave_function(xmol, u, dim[0])
					df_dy = d_wave_function(ymol, v, dim[1])
					ddf_ddx = dd_wave_function(xmol, u, dim[0])
					ddf_ddy = dd_wave_function(ymol, v, dim[1])

					temp_int_z_mol[0] += f_x * f_y * coeff[0][j]
					temp_int_z_mol[1] += f_x * f_y * coeff[1][j]
					temp_dxdy_mol[0] += df_dx * f_y * coeff[0][j]
					temp_dxdy_mol[1] += f_x * df_dy * coeff[0][j]
					temp_dxdy_mol[2] += df_dx * f_y * coeff[1][j]
					temp_dxdy_mol[3] += f_x * df_dy * coeff[1][j]
					temp_ddxddy_mol[0] += ddf_ddx * f_y * coeff[0][j]
					temp_ddxddy_mol[1] += f_x * ddf_ddy * coeff[0][j]
					temp_ddxddy_mol[2] += ddf_ddx * f_y * coeff[1][j]
					temp_ddxddy_mol[3] += f_x * ddf_ddy * coeff[1][j]

		int_z_mol[qu] += temp_int_z_mol
		int_dxdy_mol[qu] += temp_dxdy_mol
		int_ddxddy_mol[qu] += temp_ddxddy_mol

	int_z_mol = np.swapaxes(int_z_mol, 0, 1)
	int_dxdy_mol = np.swapaxes(int_dxdy_mol, 0, 1)
	int_ddxddy_mol = np.swapaxes(int_ddxddy_mol, 0, 1)
	
	return int_z_mol, int_dxdy_mol, int_ddxddy_mol


def create_intrinsic_positions_dxdyz(directory, file_name, nmol, nframe, qm, n0, phi, dim, recon=False, ow_pos=False):
	"""
	create_intrinsic_positions_dxdyz(directory, file_name, nmol, nframe, qm, n0, phi, dim, recon, ow_pos)

	Calculate distances and derivatives at each molecular position with respect to intrinsic surface in simulation frame

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed.
	nmol:  int
		Number of molecules in simulation
	nframe:  int
		Number of frames in simulation trajectory
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	n0:  int
		Maximum number of molecular pivots in intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	recon:  bool (optional)
		Whether to use surface reconstructe coefficients
	ow_pos:  bool (optional)
		Whether to overwrite positions and derivatives (default=False)

	"""

	print"\n--- Running Intrinsic Positions and Derivatives Routine ---\n"

	n_waves = 2 * qm + 1
	
	surf_dir = directory + 'surface/'
	pos_dir = directory + 'pos/'
	intpos_dir = directory + 'intpos/'
	if not os.path.exists(intpos_dir): os.mkdir(intpos_dir)

	file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	file_name_pos = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	if recon: 
		file_name_coeff += '_R'
		file_name_pos += '_R'

	if not os.path.exists('{}/{}_int_z_mol.hdf5'.format(intpos_dir, file_name_pos)):
		ut.make_hdf5(intpos_dir + file_name_pos + '_int_z_mol', (2, qm+1, nmol), tables.Float64Atom())
		ut.make_hdf5(intpos_dir + file_name_pos + '_int_dxdy_mol', (4, qm+1, nmol), tables.Float64Atom())
		ut.make_hdf5(intpos_dir + file_name_pos + '_int_ddxddy_mol', (4, qm+1, nmol), tables.Float64Atom())
		file_check = False

	elif not ow_pos:
		"Checking number of frames in current distance files"
		try:
			file_check = (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_z_mol') == (nframe, 2, qm+1, nmol))
			file_check *= (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_dxdy_mol') == (nframe, 4, qm+1, nmol))
			file_check *= (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_ddxddy_mol') == (nframe, 4, qm+1, nmol))
		except: file_check = False
	else: file_check = False

	if not file_check:
		xmol = ut.load_npy(pos_dir + file_name + '_{}_xmol'.format(nframe), frames=range(nframe))
		ymol = ut.load_npy(pos_dir + file_name + '_{}_ymol'.format(nframe), frames=range(nframe))

		for frame in xrange(nframe):

			"Checking number of frames in int_z_mol file"
			frame_check_int_z_mol = (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_z_mol')[0] <= frame)
			frame_check_int_dxdy_mol = (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_dxdy_mol')[0] <= frame)
			frame_check_int_ddxddy_mol = (ut.shape_check_hdf5(intpos_dir + file_name_pos + '_int_ddxddy_mol')[0] <= frame)

			if frame_check_int_z_mol: mode_int_z_mol = 'a'
			elif ow_pos: mode_int_z_mol = 'r+'
			else: mode_int_z_mol = False

			if frame_check_int_dxdy_mol: mode_int_dxdy_mol = 'a'
			elif ow_pos: mode_int_dxdy_mol = 'r+'
			else: mode_int_dxdy_mol = False

			if frame_check_int_ddxddy_mol: mode_int_ddxddy_mol = 'a'
			elif ow_pos: mode_int_ddxddy_mol = 'r+'
			else: mode_int_ddxddy_mol = False

			if not mode_int_z_mol and not mode_int_dxdy_mol and not mode_int_ddxddy_mol: pass
			else:
				sys.stdout.write("Calculating molecular distances and derivatives: frame {}\r".format(frame))
				sys.stdout.flush()
			
				coeff = ut.load_hdf5(surf_dir + file_name_coeff + '_coeff', frame)

				int_z_mol, int_dxdy_mol, int_ddxddy_mol = make_pos_dxdy(directory, file_name_pos, xmol[frame], ymol[frame], coeff, nmol, dim, qm)
				ut.save_hdf5(intpos_dir + file_name_pos + '_int_z_mol', int_z_mol, frame, mode_int_z_mol)
				ut.save_hdf5(intpos_dir + file_name_pos + '_int_dxdy_mol', int_dxdy_mol, frame, mode_int_dxdy_mol)
				ut.save_hdf5(intpos_dir + file_name_pos + '_int_ddxddy_mol', int_ddxddy_mol, frame, mode_int_ddxddy_mol)


def make_den_curve(directory, zmol, int_z_mol, int_dxdy_mol, coeff, nmol, nslice, nz, qm, dim):
	"""
	make_den_curve(directory, zmol, int_z_mol, int_dxdy_mol, coeff, nmol, nslice, nz, qm, dim)

	Creates density and curvature distributions normal to surface

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	zmol:  float, array_like; shape=(nmol)
		Molecular coordinates in z dimension
	int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
		Molecular distances from intrinsic surface
	int_dxdy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
		First derivatives of intrinsic surface wrt x and y at xmol, ymol
	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	nmol:  int
		Number of molecules in simulation
	nslice: int
		Number of bins in density histogram along axis normal to surface
	nz: int (optional)
		Number of bins in curvature histogram along axis normal to surface (default=100)
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	count_corr_array:  int, array_like; shape=(qm+1, nslice, nz)
		Number histogram binned by molecular position along z axis and mean curvature H across qm resolutions 

	"""

	lslice = dim[2] / nslice

	count_corr_array = np.zeros((qm+1, nslice, nz))

	for qu in xrange(qm+1):

		temp_count_corr_array = np.zeros((nslice, nz))

		int_z1 = int_z_mol[0][qu]
		int_z2 = int_z_mol[1][qu]

		z1 = zmol - int_z1
		z2 = -zmol + int_z2

		dzx1 = int_dxdy_mol[0][qu]
		dzy1 = int_dxdy_mol[1][qu]
		dzx2 = int_dxdy_mol[2][qu]
		dzy2 = int_dxdy_mol[3][qu]

		index1_mol = np.array((z1 + dim[2]/2.) * nslice / dim[2], dtype=int) % nslice
		index2_mol = np.array((z2 + dim[2]/2.) * nslice / dim[2], dtype=int) % nslice

		normal1 = ut.unit_vector(np.array([-dzx1, -dzy1, np.ones(nmol)]))
		normal2 = ut.unit_vector(np.array([-dzx2, -dzy2, np.ones(nmol)]))

		index1_nz = np.array(abs(normal1[2]) * nz, dtype=int) % nz
		index2_nz = np.array(abs(normal2[2]) * nz, dtype=int) % nz

		temp_count_corr_array += np.histogram2d(index1_mol, index1_nz, bins=[nslice, nz], range=[[0, nslice], [0, nz]])[0]
		temp_count_corr_array += np.histogram2d(index2_mol, index2_nz, bins=[nslice, nz], range=[[0, nslice], [0, nz]])[0]

		count_corr_array[qu] += temp_count_corr_array

	return count_corr_array


def create_intrinsic_den_curve_dist(directory, file_name, qm, n0, phi, nframe, nslice, dim, nz=100, recon=False, ow_hist=False):
	"""
	create_intrinsic_den_curve_dist(directory, file_name, qm, n0, phi, nframe, nslice, dim, nz=100, nnz=100, recon=False, ow_count=False)

	Calculate density and curvature distributions across surface

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed.
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	n0:  int
		Maximum number of molecular pivots in intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	nframe:  int
		Number of frames in simulation trajectory
	nslice: int
		Number of bins in density histogram along axis normal to surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell
	nz: int (optional)
		Number of bins in curvature histogram along axis normal to surface (default=100)
	recon:  bool (optional)
		Whether to use surface reconstructe coefficients (default=False)
	ow_count:  bool (optional)
		Whether to overwrite density and curvature distributions (default=False)
	"""

	print"\n--- Running Intrinsic Density and Curvature Routine --- \n"

	surf_dir = directory + 'surface/'
	pos_dir = directory + 'pos/'
	intpos_dir = directory + 'intpos/'
	intden_dir = directory + 'intden/'
	if not os.path.exists(intden_dir): os.mkdir(intden_dir)

	lslice = dim[2] / nslice

	file_name_pos = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
	file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
	file_name_hist = '{}_{}_{}_{}_{}_{}_{}'.format(file_name, nslice, nz, qm, n0, int(1./phi + 0.5), nframe)	

	if recon:
		file_name_pos += '_R'
		file_name_hist += '_R'
		file_name_coeff += '_R'

	if not os.path.exists(intden_dir + file_name_hist + '_count_corr.hdf5'):
		ut.make_hdf5(intden_dir + file_name_hist + '_count_corr', (qm+1, nslice, nz), tables.Float64Atom())
		file_check = False

	elif not ow_hist:
		"Checking number of frames in current distribution files"
		try: file_check = (ut.shape_check_hdf5(intden_dir + file_name_hist + '_count_corr') == (nframe, qm+1, nslice, nz))
		except: file_check = False
	else:file_check = False

	if not file_check:
		zmol = ut.load_npy(pos_dir + file_name + '_{}_zmol'.format(nframe))
		#COM = ut.load_npy(pos_dir + file_name + '_{}_com'.format(nframe))
		nmol = zmol.shape[1]
		#com_tile = np.moveaxis(np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
		#zmol = zmol - com_tile

		for frame in xrange(nframe):

			"Checking number of frames in hdf5 files"
			frame_check_count_corr = (ut.shape_check_hdf5(intden_dir + file_name_hist + '_count_corr') <= frame)

			if frame_check_count_corr: mode_count_corr = 'a'
			elif ow_hist: mode_count_corr = 'r+'
			else: mode_count_corr = False

			if not mode_count_corr:pass
			else:
				sys.stdout.write("Calculating position and curvature distributions: frame {}\r".format(frame))
				sys.stdout.flush()

				coeff = ut.load_hdf5(surf_dir + file_name_coeff + '_coeff', frame)
				int_z_mol = ut.load_hdf5(intpos_dir + file_name_pos + '_int_z_mol', frame)
				int_dxdy_mol = ut.load_hdf5(intpos_dir + file_name_pos + '_int_dxdy_mol', frame)

				count_corr_array = make_den_curve(directory, zmol[frame], int_z_mol, int_dxdy_mol, coeff, nmol, nslice, nz, qm, dim)
				ut.save_hdf5(intden_dir + file_name_hist + '_count_corr', count_corr_array, frame, mode_count_corr)



