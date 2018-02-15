"""
*************** INTRINSIC SAMPLING METHOD MODULE *******************

Performs intrinsic sampling analysis on a set of interfacial 
simulation configurations

********************************************************************
Created 24/11/16 by Frank Longford

Last modified 14/12/18 by Frank Longford
"""

import numpy as np
import scipy as sp

import utilities as ut

import os, sys, time, tables

sqrt_2 = np.sqrt(2.)


def check_uv(u, v):
	"""
	check_uv(u, v)

	Returns weightings for frequencies u and v for anisotropic surfaces
	
	"""

	if abs(u) + abs(v) == 0: return 1.
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
		Matrix containing weightings for each coefficient product x_i.x_j in the linear algebra equation Ax = b for both surfaces 
	b:  float, array_like; shape=(2, n_waves**2)
		Vector containing solutions to the linear algebra equation Ax = b for both surfaces

	"""

	A = np.zeros((2, n_waves**2, n_waves**2))
	b = np.zeros((2, n_waves**2))

	fuv1 = np.zeros((n_waves**2, len(new_piv1)))
	fuv2 = np.zeros((n_waves**2, len(new_piv2)))

	for j in xrange(n_waves**2):
		fuv1[j] = wave_function(xmol[new_piv1], int(j/n_waves)-qm, dim[0]) * wave_function(ymol[new_piv1], int(j%n_waves)-qm, dim[1])
		b[0][j] += np.sum(zmol[new_piv1] * fuv1[j])
		fuv2[j] = wave_function(xmol[new_piv2], int(j/n_waves)-qm, dim[0]) * wave_function(ymol[new_piv2], int(j%n_waves)-qm, dim[1])
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

	A:  float, array_like; shape=(n_waves**2, n_waves**2)
		Matrix containing weightings for each coefficient product x_i.x_j in the linear algebra equation Ax = b
	b:  float, array_like; shape=(n_waves**2)
		Vector containing solutions to the linear algebra equation Ax = b

	Returns
	-------

	coeff:	array_like (float); shape=(n_waves**2)
		Optimised surface coefficients	

	"""
	lu, piv  = sp.linalg.lu_factor(A)
	coeff = sp.linalg.lu_solve((lu, piv), b)

	return coeff


def make_zeta_list(xmol, ymol, dim, mol_list, coeff, qm):
	"""
	zeta_list(xmol, ymol, dim, mol_list, coeff, qm)

	Calculate dz (zeta) between molecular sites and intrinsic surface for highest resolution"

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

	zeta_list = chi(xmol[mol_list], ymol[mol_list], coeff, qm, qm, dim)
   
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
		Indices of molcules available to be slected as pivots
	zeta_list:  float, array_like; shape=(n0)
		Array of dz (zeta) between molecular sites and intrinsic surface
	piv_n:  int, array_like; shape=(n0)
		Molecular pivot indices
	tau:  float
		Threshold length along z axis either side of existing intrinsic surface for selection of new pivot points

	Returns
	-------

	mol_list:  int, array_like; shape=(n0)
		Updated indices of molcules available to be slected as pivots
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
	ut.bubblesort(new_piv, dz_new_piv)

	"Add new pivots to pivoy list and check whether max n0 pivots are selected"
	piv_n = np.append(piv_n, new_piv)
	if len(piv_n) > n0: 
		new_piv = new_piv[:len(piv_n)-n0]
		piv_n = piv_n[:n0] 

	far_tau = 6.0 * tau
	
	"Remove pivots far form molecular search list"
	far_piv = mol_list[zeta > far_tau]
	if len(new_piv) > 0: mol_list = ut.numpy_remove(mol_list, np.append(new_piv, far_piv))
	
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

	coeff_2 = coeff**2

	int_A = np.pi**2  * vcheck(u_array, v_array) * (u_array**2 / dim[0]**2 + v_array**2 / dim[1]**2) * coeff_2
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
	
	mat_xmol = np.tile(xmol, (nmol, 1))
	mat_ymol = np.tile(ymol, (nmol, 1))
	mat_zmol = np.tile(zmol, (nmol, 1))

	dr2 = np.array((mat_xmol - mat_xmol.T)**2 + (mat_ymol - mat_ymol.T)**2 + (mat_zmol - mat_zmol.T)**2, dtype=float)
	
	"Remove molecules from vapour phase ans assign an initial grid of pivots furthest away from centre of mass"
	print 'Lx = {:5.3f}   Ly = {:5.3f}   qm = {:5d}\nphi = {}   n_piv = {:5d}   vlim = {:5d}   max_r = {:5.3f}'.format(dim[0], dim[1], qm, phi, n0, vlim, max_r) 
	print 'Selecting initial {} pivots'.format(ncube**2)

	for n in xrange(nmol):
		vapour = np.count_nonzero(dr2[n] < max_r**2) - 1
		if vapour > vlim:

			indexx = int(xmol[n] * ncube / dim[0]) % ncube
                        indexy = int(ymol[n] * ncube / dim[1]) % ncube

			if zmol[n] < piv_z1[ncube*indexx + indexy]: 
				piv_n1[ncube*indexx + indexy] = n
				piv_z1[ncube*indexx + indexy] = zmol[n]
			elif zmol[n] > piv_z2[ncube*indexx + indexy]: 
				piv_n2[ncube*indexx + indexy] = n
				piv_z2[ncube*indexx + indexy] = zmol[n]

		else: vapour_list.append(n)

	"Update molecular and pivot lists"
	mol_list = ut.numpy_remove(mol_list, vapour_list)
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

	n_waves = 2*qm+1

	"Form the diagonal chi^2 terms"
	u = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
        v = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

	diag = vcheck(u, v) * (u**2 * dim[1] / dim[0] + v**2 * dim[0] / dim[1])
	diag = 4 * np.pi**2 * phi * np.diagflat(diag)

	"Create empty A matrix and b vector for linear algebra equation Ax = b"
	A = np.zeros((2, n_waves**2, n_waves**2))
	b = np.zeros((2, n_waves**2))

	print "{:^77s} | {:^43s} | {:^21s} | {:^21s}".format('TIMINGS (s)', 'PIVOTS', 'TAU', 'INT AREA')
	print ' {:20s}  {:20s}  {:20s}  {:10s} | {:10s} {:10s} {:10s} {:10s} | {:10s} {:10s} | {:10s} {:10s}'.format('Matrix Formation', 'LU Decomposition', 'Pivot selection', 'TOTAL', 'n_piv1', '(new)', 'n_piv2', '(new)', 'surf1', 'surf2', 'surf1', 'surf2')
	print "_" * 170

	building_surface = True
	build_surf1 = True
	build_surf2 = True

	coeff = np.zeros((2, n_waves**2))

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
		if build_surf1: zeta_list1 = make_zeta_list(xmol, ymol, dim, mol_list1, coeff[0], qm)
		if build_surf2: zeta_list2 = make_zeta_list(xmol, ymol, dim, mol_list2, coeff[1], qm)

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

	return coeff, pivot


def H_var_coeff(coeff, qm, qu, dim):
	"""
	H_var_coeff(coeff, qm, qu, dim)

	Variance of mean curvature H across surface determined by coeff at resolution qu

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

	H_var:  float
		Variance of mean curvature H across whole surface

	"""

	if qu == 0: return 0
	
	n_waves = 2 * qm +1
	
	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

	H_var = 4 * np.pi**4 * vcheck(u_array, v_array) * coeff**2
	H_var *= (u_array**4 / dim[0]**4 + v_array**4 / dim[1]**4 + 2 * u_array**2 * v_array**2 / (dim[0]**2 * dim[1]**2))
	H_var *= (u_array >= -qu) * (u_array <= qu) * (v_array >= -qu) * (v_array <= qu)

	return np.sum(H_var)


def H_var_piv(xmol, ymol, coeff, pivot, qm, qu, dim):
	"""
	H_var_piv(xmol, ymol, coeff, pivot, qm, qu, dim)

	Variance of mean curvature H at pivot points determined by coeff at resolution qu

	Parameters
	----------

	xmol:  float, array_like; shape=(nmol)
		Molecular coordinates in x dimension
	ymol:  float, array_like; shape=(nmol)
		Molecular coordinates in y dimension
	coeff:	float, array_like; shape=(n_waves**2)
		Optimised surface coefficients
	pivot:  int, array_like; shape=(2, n0)
		Indicies of pivot molecules in molecular position arrays
	qm:  int
		Maximum number of wave frequencies in Fouier Sum representing intrinsic surface
	qu:  int
		Upper limit of wave frequencies in Fouier Sum representing intrinsic surface
	dim:  float, array_like; shape=(3)
		XYZ dimensions of simulation cell

	Returns
	-------

	H_var:  float
		Variance of mean curvature H at pivot points

	"""

	if qu == 0: return 0
	
	n_waves = 2 * qm +1
	n0 = len(pivot)

	"Create arrays of wave frequency indicies u and v"
	u_array = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v_array = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

	"Cancel terms of higher resolution than qu"
	u_array *= (u_array >= -qu) * (u_array <= qu)
	v_array *= (v_array >= -qu) * (v_array <= qu)

	"Create matrix of wave frequency indicies (u,v)**2"
	u_matrix = np.tile(u_array, (n_waves**2, 1))
	v_matrix = np.tile(v_array, (n_waves**2, 1))

	"Make curvature diagonal terms of A matrix"
	curve_diag = 16 * np.pi**4 * (u_matrix**2 * u_matrix.T**2 / dim[0]**4 + v_matrix**2 * v_matrix.T**2 / dim[1]**4 +
				     (u_matrix**2 * v_matrix.T**2 + u_matrix.T**2 * v_matrix**2) / (dim[0]**2 * dim[1]**2))

	"Form the diagonal chi^2 terms and b vector solutions"
        fuv = np.zeros((n_waves**2, n0))
        for j in xrange(n_waves**2):
                fuv[j] = wave_function(xmol[pivot], int(j/n_waves)-qm, dim[0]) * wave_function(ymol[pivot], int(j%n_waves)-qm, dim[1])
	ffuv = np.dot(fuv, fuv.T)

	coeff_matrix = np.tile(coeff, (n_waves**2, 1))
	H_var = np.sum(coeff_matrix * coeff_matrix.T * ffuv * curve_diag / n0)

	return np.sum(H_var)


def surface_reconstruction(coeff, pivot, xmol, ymol, zmol, dim, qm, n0, phi, psi):
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

	var_lim = 1E-3
	n_waves = 2*qm+1

	print "PERFORMING SURFACE RESTRUCTURING"
	print 'Lx = {:5.3f}   Ly = {:5.3f}   qm = {:5d}\nphi = {}  psi = {}  n_piv = {:5d}  var_lim = {}'.format(dim[0], dim[1], qm, phi, psi, n0, var_lim) 
	print 'Setting up wave product and coefficient matricies'

	orig_psi1 = psi
	orig_psi2 = psi
	psi1 = psi
	psi2 = psi

	start = time.time()	

	"Form the diagonal chi^2 terms and b vector solutions"
	A = np.zeros((2, n_waves**2, n_waves**2))
	fuv1 = np.zeros((n_waves**2, n0))
        fuv2 = np.zeros((n_waves**2, n0))
	b = np.zeros((2, n_waves**2))

        for j in xrange(n_waves**2):
                fuv1[j] = wave_function(xmol[pivot[0]], int(j/n_waves)-qm, dim[0]) * wave_function(ymol[pivot[0]], int(j%n_waves)-qm, dim[1])
                b[0][j] += np.sum(zmol[pivot[0]] * fuv1[j])
                fuv2[j] = wave_function(xmol[pivot[1]], int(j/n_waves)-qm, dim[0]) * wave_function(ymol[pivot[1]], int(j%n_waves)-qm, dim[1])
                b[1][j] += np.sum(zmol[pivot[1]] * fuv2[j])

	ffuv1 = np.dot(fuv1, fuv1.T)
	ffuv2 = np.dot(fuv2, fuv2.T)

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

	print "{:^74s} | {:^21s} | {:^43s}".format('TIMINGS (s)', 'PSI', 'VAR(H)' )
	print ' {:20s} {:20s} {:20s} {:10s} | {:10s} {:10s} | {:10s} {:10s} {:10s} {:10s}'.format('Matrix Formation', 'LU Decomposition', 'Var Estimation', 'TOTAL', 'surf1', 'surf2', 'surf1', 'piv1', 'surf2', 'piv2')
	print "_" * 165

	"Calculate variance of curvature across entire surface from coefficients"

	H_var = 4 * np.pi**4 * uv_check * (u_array**4 / dim[0]**4 + v_array**4 / dim[1]**4 + 2 * u_array**2 * v_array**2 / (dim[0]**2 * dim[1]**2))
	H_var_coeff1 = np.sum(H_var * coeff[0]**2)
	H_var_coeff2 = np.sum(H_var * coeff[1]**2)

	"Calculate variance of curvature at pivot sites only"
	coeff1_matrix = np.tile(coeff[0], (n_waves**2, 1))
	H_var_piv1 = np.sum(coeff1_matrix * coeff1_matrix.T * ffuv1 * curve_diag / n0)

	coeff2_matrix = np.tile(coeff[1], (n_waves**2, 1))
	H_var_piv2 = np.sum(coeff2_matrix * coeff2_matrix.T * ffuv2 * curve_diag / n0)

	end_setup2 = time.time()

	print ' {:20.3f} {:20.3f} {:20.3f} {:10.3f} | {:10.6f} {:10.6f} | {:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(end_setup1-start, 
			0, end_setup2-end_setup1, end_setup2-start, 0, 0, H_var_coeff1, H_var_piv1, H_var_coeff2, H_var_piv2)

	coeff_recon = np.zeros(coeff.shape)
	reconstructing = True
	recon_1 = True
	recon_2 = True
	loop1 = 0
	loop2 = 0

	"Amend psi weighting coefficient until H_var == H_piv_var"
	while reconstructing:

		start1 = time.time()
        
		"Update A matrix and b vector"
		A[0] = ffuv1 * (1. + curve_diag * psi1 / n0)
		A[1] = ffuv2 * (1. + curve_diag * psi2 / n0) 

		end1 = time.time()

		"Perform LU decomosition to solve Ax = b"
		if recon_1: coeff_recon[0] = LU_decomposition(A[0] + diag, b[0])
		if recon_2: coeff_recon[1] = LU_decomposition(A[1] + diag, b[1])

		end2 = time.time()

		"Recalculate variance of curvature across entire surface from coefficients"
		H_var_coeff1_recon = np.sum(H_var * coeff_recon[0]**2)
		H_var_coeff2_recon = np.sum(H_var * coeff_recon[1]**2)

		"Recalculate variance of curvature at pivot sites only"
		if recon_1:
			coeff1_matrix_recon = np.tile(coeff_recon[0], (n_waves**2, 1))
			H_var_piv1_recon = np.sum(coeff1_matrix_recon * coeff1_matrix_recon.T * ffuv1 * curve_diag / n0)
		if recon_2:
			coeff2_matrix_recon = np.tile(coeff_recon[1], (n_waves**2, 1))
			H_var_piv2_recon = np.sum(coeff2_matrix_recon * coeff2_matrix_recon.T * ffuv2 * curve_diag / n0)

		end3 = time.time()

		print ' {:20.3f} {:20.3f} {:20.3f} {:10.3f} | {:10.6f} {:10.6f} | {:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(end1 - start1, 
				end2 - end1, end3 - end2, end3 - start1, psi1, psi2, H_var_coeff1, H_var_coeff1_recon, H_var_coeff2, H_var_coeff2_recon)

		if abs(H_var_piv1_recon - H_var_coeff1_recon) <= var_lim: recon_1 = False
		else: 
			psi1 += orig_psi1 * (H_var_piv1_recon - H_var_coeff1_recon)
			if abs(H_var_piv1_recon) > 5 * H_var_coeff1 or loop1 > 40:
				orig_psi1 *= 0.5 
				psi1 = orig_psi1
				loop1 = 0
			else: loop1 += 1
		if abs(H_var_piv2_recon - H_var_coeff2_recon) <= var_lim: recon_2 = False
		else: 
			psi2 += orig_psi2 * (H_var_piv2_recon - H_var_coeff2_recon)
			if abs(H_var_piv2_recon) > 5 * H_var_coeff2 or loop2 > 40: 
				orig_psi2 *= 0.5 
				psi2 = orig_psi2
				loop2 = 0
			else: loop2 += 1

		if not recon_1 and not recon_2: reconstructing = False

	end = time.time()

	print '\nTOTAL time: {:7.2f} s \n'.format(end - start)

	return coeff_recon


def chi(x, y, coeff, qm, qu, dim):
	"""
	chi(x, y, coeff, qm, qu, dim)

	Function returning position of intrinsic surface at position (x,y)

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

	chi_z:  float
		Position of intrinsic surface in z dimension

	"""
	
	chi_z = 0

	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			chi_z += wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]

	return chi_z


def dxy_dchi(x, y, coeff, qm, qu, dim):
	"""
	dxy_dchi(x, y, qm, qu, coeff, dim)

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

	dx_dchi:  float
		Derivative of intrinsic surface in x dimension
	dy_dchi:  float
		Derivative of intrinsic surface in y dimension
	
	"""
	dx_dchi = 0
	dy_dchi = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			dx_dchi += d_wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
			dy_dchi += wave_function(x, u, dim[0]) * d_wave_function(y, v, dim[1]) * coeff[j]

	return dx_dchi, dy_dchi


def ddxy_ddchi(x, y, coeff, qm, qu, dim):
	"""
	ddxy_ddchi(x, y, coeff, qm, qu, dim)

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

	ddx_ddchi:  float
		Second derivative of intrinsic surface in x dimension
	ddy_ddchi:  float
		Second derivative of intrinsic surface in y dimension
	
	"""
	ddx_ddchi = 0
	ddy_ddchi = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			ddx_ddchi += dd_wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
			ddy_ddchi += wave_function(x, u, dim[0]) * dd_wave_function(y, v, dim[1]) * coeff[j]

	return ddx_ddchi, ddy_ddchi


def H_xy(x, y, coeff, qm, qu, dim):
	"""
	H_xy(x, y, coeff, qm, qu, dim)

	Calculation of mean curvature at position (x,y) at resolution qu

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

	dx_dchi:  float
		Derivative of intrinsic surface in x dimension
	dy_dchi:  float
		Derivative of intrinsic surface in y dimension
	"""

	H = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			H += -4 * np.pi**2 * (u**2 / dim[0]**2 + v**2 / dim[1]**2) * wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * coeff[j]
	return H


def optimise_ns(directory, file_name, nmol, nframe, qm, phi, dim, mol_sigma, start_ns, step_ns, nframe_ns = 20):
	"""
	optimise_ns(directory, file_name, nmol, nframe, qm, phi, vlim, ncube, dim, mol_sigma, start_ns, step_ns, nframe_ns = 20)

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

	if not os.path.exists("{}/surface".format(directory)): os.mkdir("{}/surface".format(directory))

	mol_ex_1 = []
	mol_ex_2 = []
	NS = []

	n_waves = 2 * qm + 1
	max_r = 1.5 * mol_sigma
	tau = 0.5 * mol_sigma

	xmol, ymol, zmol = ut.read_mol_positions(directory, file_name, nframe, nframe_ns)
	COM = ut.read_com_positions(directory, file_name, nframe, nframe_ns)
	com_tile = np.moveaxis(np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
	zmol = zmol - com_tile

	if nframe < nframe_ns: nframe_ns = nframe
	ns = start_ns
	optimising = True

	surf_dir = directory + '/surface'

	while optimising:

		NS.append(ns)
		n0 = int(dim[0] * dim[1] * ns / mol_sigma**2)

		tot_piv_n1 = np.zeros((nframe_ns, n0))
		tot_piv_n2 = np.zeros((nframe_ns, n0))

		file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)

		if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_coeff)):
			ut.make_hdf5(surf_dir, file_name_coeff + '_coeff', (2, n_waves**2), tables.Float64Atom())
			ut.make_hdf5(surf_dir, file_name_coeff + '_pivot', (2, n0), tables.Int64Atom())

		for frame in xrange(nframe_ns):
			"Checking number of frames in coeff and pivot files"
			frame_check_coeff = (ut.shape_check_hdf5(surf_dir, file_name_coeff + '_coeff')[0] <= frame)
			frame_check_pivot = (ut.shape_check_hdf5(surf_dir, file_name_coeff + '_pivot')[0] <= frame)

			if frame_check_coeff: mode_coeff = 'a'
			elif ow_coeff: mode_coeff = 'r+'
			else: mode_coeff = False

			if frame_check_pivot: mode_pivot = 'a'
			elif ow_coeff: mode_pivot = 'r+'
			else: mode_pivot = False

			if not mode_coeff and not mode_pivot:
				pivot = ut.load_hdf5(surf_dir, file_name_coeff + '_pivot', frame)
			else:
				sys.stdout.write("Optimising Intrinsic Surface coefficients: frame {}\n".format(frame))
				sys.stdout.flush()

				coeff, pivot = build_surface(xmol[frame], ymol[frame], zmol[frame], dim, mol_sigma, qm, n0, phi, tau, max_r)
				ut.save_hdf5(surf_dir, file_name_coeff + '_coeff', coeff, frame, mode_coeff)
				ut.save_hdf5(surf_dir, file_name_coeff + '_pivot', pivot, frame, mode_pivot)

			tot_piv_n1[frame] += pivot[0]
			tot_piv_n2[frame] += pivot[1]

		ex_1, ex_2 = mol_exchange(tot_piv_n1, tot_piv_n2, nframe_ns, n0)

		mol_ex_1.append(ex_1)
		mol_ex_2.append(ex_2)

		if len(mol_ex_1) > 1:
			check = np.argmin((np.array(mol_ex_1) + np.array(mol_ex_2)) / 2.) == (len(NS) - 1)
			if not check: optimising = False
			else: 
				ns += step_ns
				print "Optimal surface density not found.\nContinuing search using pivot number = {}".format(int(dim[0] * dim[1] * ns / mol_sigma**2))
		else: ns += step_ns

	opt_ns = NS[np.argmin((np.array(mol_ex_1) + np.array(mol_ex_2)) / 2.)]
	opt_n0 = int(dim[0] * dim[1] * opt_ns / mol_sigma**2)

	print "Optimal pivot density found = {}".format(opt_n0)

	for ns in NS:
		if ns != opt_ns:
			n0 = int(dim[0] * dim[1] * ns / mol_sigma**2)
			file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
			os.remove('{}/surface/{}_coeff.hdf5'.format(directory, file_name_coeff))
			os.remove('{}/surface/{}_pivot.hdf5'.format(directory, file_name_coeff))

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



def create_intrinsic_surfaces(directory, file_name, dim, qm, n0, phi, mol_sigma, nframe, recon=True, ow_coeff=False, ow_recon=False):
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

	print"\n-- Running Intrinsic Surface Routine ---\n"

	if not os.path.exists("{}/surface".format(directory)): os.mkdir("{}/surface".format(directory))

	surf_dir = directory + '/surface'
	file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	n_waves = 2 * qm + 1
	max_r = 1.5 * mol_sigma
	tau = 0.5 * mol_sigma


	"Make coefficient and pivot files"
	if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_coeff)):
		ut.make_hdf5(surf_dir, file_name_coeff + '_coeff', (2, n_waves**2), tables.Float64Atom())
		ut.make_hdf5(surf_dir, file_name_coeff + '_pivot', (2, n0), tables.Int64Atom())
		file_check = False
	elif not ow_coeff:
		"Checking number of frames in current coefficient files"
		try:
			file_check = (ut.shape_check_hdf5(surf_dir, file_name_coeff + '_coeff') == (nframe, 2, n_waves**2))
			file_check *= (ut.shape_check_hdf5(surf_dir, file_name_coeff + '_pivot') == (nframe, 2, n0))
		except: file_check = False
	else: file_check = False

	if recon:
		psi = phi * dim[0] * dim[1]
		"Make recon coefficient file"
		if not os.path.exists('{}/surface/{}_R_coeff.hdf5'.format(directory, file_name_coeff)):
			ut.make_hdf5(surf_dir, file_name_coeff + '_R_coeff', (2, n_waves**2), tables.Float64Atom())
			file_check = False
		elif not ow_recon:
			"Checking number of frames in current recon coefficient files"
			try: file_check = (ut.shape_check_hdf5(surf_dir, file_name_coeff + '_R_coeff') == (nframe, 2, n_waves**2))
			except: file_check = False
		else: file_check = False

	if not file_check:
		print "IMPORTING GLOBAL POSITION DISTRIBUTIONS\n"
		xmol, ymol, zmol = ut.read_mol_positions(directory, file_name, nframe, nframe)
		COM = ut.read_com_positions(directory, file_name, nframe, nframe)
		nmol = xmol.shape[1]
		com_tile = np.moveaxis(np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]

		assert com_tile.shape == (nframe, nmol) 

		zmol = zmol - com_tile

		for frame in xrange(nframe):
			"Checking number of frames in coeff and pivot files"
			frame_check_coeff = (ut.shape_check_hdf5(surf_dir, file_name_coeff + '_coeff')[0] <= frame)
			frame_check_pivot = (ut.shape_check_hdf5(surf_dir, file_name_coeff + '_pivot')[0] <= frame)

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

				coeff, pivot = build_surface(xmol[frame], ymol[frame], zmol[frame], dim, mol_sigma, qm, n0, phi, tau, max_r)
				ut.save_hdf5(surf_dir, file_name_coeff + '_coeff', coeff, frame, mode_coeff)
				ut.save_hdf5(surf_dir, file_name_coeff + '_pivot', pivot, frame, mode_pivot)

			if recon:
				frame_check_coeff = (ut.shape_check_hdf5(surf_dir, file_name_coeff + '_R_coeff')[0] <= frame)

				if frame_check_coeff: mode_coeff = 'a'
				elif ow_coeff: mode_coeff = 'r+'
				else: mode_coeff = False

				if not mode_coeff: pass
				else:
					sys.stdout.write("Reconstructing Intrinsic Surface coefficients: frame {}\r".format(frame))
					sys.stdout.flush()

					coeff = ut.load_hdf5(surf_dir, file_name_coeff + '_coeff', frame)
					pivot = ut.load_hdf5(surf_dir, file_name_coeff + '_pivot', frame)
					coeff_R = surface_reconstruction(coeff, pivot, xmol[frame], ymol[frame], zmol[frame], dim, qm, n0, phi, psi)
					ut.save_hdf5(surf_dir, file_name_coeff + '_R_coeff', coeff_R, frame, mode_coeff)


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

	print"\n-- Running Intrinsic Positions and Derivatives Routine ---\n"

	n_waves = 2 * qm + 1
	
	surf_dir = directory + '/surface'
	pos_dir = directory + '/intpos'
	if not os.path.exists(pos_dir): os.mkdir(pos_dir)

	file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	if recon: file_name_coeff += '_R'

	file_name_pos = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	if recon: file_name_pos += '_R'

	if not os.path.exists('{}/intpos/{}_int_z_mol.hdf5'.format(directory, file_name_pos)):
		ut.make_hdf5(pos_dir, file_name_pos + '_int_z_mol', (2, qm+1, nmol), tables.Float64Atom())
		ut.make_hdf5(pos_dir, file_name_pos + '_int_dxdy_mol', (4, qm+1, nmol), tables.Float64Atom())
		ut.make_hdf5(pos_dir, file_name_pos + '_int_ddxddy_mol', (4, qm+1, nmol), tables.Float64Atom())
		file_check = False

	elif not ow_pos:
		"Checking number of frames in current distance files"
		try:
			file_check = (ut.shape_check_hdf5(pos_dir, file_name_pos + '_int_z_mol') == (nframe, 2, qm+1, nmol))
			file_check *= (ut.shape_check_hdf5(pos_dir, file_name_pos + '_int_dxdy_mol') == (nframe, 4, qm+1, nmol))
			file_check *= (ut.shape_check_hdf5(pos_dir, file_name_pos + '_int_ddxddy_mol') == (nframe, 4, qm+1, nmol))
		except: file_check = False
	else: file_check = False

	if not file_check:
		xmol, ymol, _ = ut.read_mol_positions(directory, file_name, nframe, nframe)

		for frame in xrange(nframe):

			"Checking number of frames in int_z_mol file"
			frame_check_int_z_mol = (ut.shape_check_hdf5(pos_dir, file_name_pos + '_int_z_mol')[0] <= frame)
			frame_check_int_dxdy_mol = (ut.shape_check_hdf5(pos_dir, file_name_pos + '_int_dxdy_mol')[0] <= frame)
			frame_check_int_ddxddy_mol = (ut.shape_check_hdf5(pos_dir, file_name_pos + '_int_ddxddy_mol')[0] <= frame)

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
			
				coeff = ut.load_hdf5(surf_dir, file_name_coeff + '_coeff', frame)

				int_z_mol, int_dxdy_mol, int_ddxddy_mol = make_pos_dxdy(directory, file_name_pos, xmol[frame], ymol[frame], coeff, nmol, dim, qm)
				ut.save_hdf5(pos_dir, file_name_pos + '_int_z_mol', int_z_mol, frame, mode_int_z_mol)
				ut.save_hdf5(pos_dir, file_name_pos + '_int_dxdy_mol', int_dxdy_mol, frame, mode_int_dxdy_mol)
				ut.save_hdf5(pos_dir, file_name_pos + '_int_ddxddy_mol', int_ddxddy_mol, frame, mode_int_ddxddy_mol)


def make_den_curve(directory, zmol, int_z_mol, int_dxdy_mol, coeff, nmol, nslice, nz, nnz, qm, dim):

	lslice = dim[2] / nslice

	count_corr_array = np.zeros((qm+1, nslice, nnz))
	z_nz_array = np.zeros((qm+1, nz, nnz))	

	for qu in xrange(qm+1):

		temp_count_corr_array = np.zeros((nslice, nnz))
		temp_z_nz_array = np.zeros((nz, nnz))

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

		index1_nz = np.array(abs(normal1[2]) * nnz, dtype=int) % nnz
		index2_nz = np.array(abs(normal2[2]) * nnz, dtype=int) % nnz

		temp_count_corr_array += np.histogram2d(index1_mol, index1_nz, bins=[nslice, nnz], range=[[0, nslice], [0, nnz]])[0]
		temp_count_corr_array += np.histogram2d(index2_mol, index2_nz, bins=[nslice, nnz], range=[[0, nslice], [0, nnz]])[0]

		index1_mol = np.array(abs(int_z1 - coeff[0][len(coeff[0])/2]) * 2 * nz / (nz*lslice), dtype=int) % nz
		index2_mol = np.array(abs(int_z2 - coeff[1][len(coeff[1])/2]) * 2 * nz / (nz*lslice), dtype=int) % nz

		temp_z_nz_array += np.histogram2d(index1_mol, index1_nz, bins=[nz, nnz], range=[[0, nz], [0, nnz]])[0]
		temp_z_nz_array += np.histogram2d(index2_mol, index2_nz, bins=[nz, nnz], range=[[0, nz], [0, nnz]])[0]

		count_corr_array[qu] += temp_count_corr_array
		z_nz_array[qu] += temp_z_nz_array

	return count_corr_array, z_nz_array


def create_intrinsic_den_curve_dist(directory, file_name, qm, n0, phi, nframe, nslice, nz, nnz, dim, recon=False, ow_count=False):
	"""
	create_intrinsic_den_curve_dist(directory, file_name, qm, n0, phi, nframe, nslice, nz, nnz, dim, recon=False, ow_count=False)

	Calculate density and curvature distributions across surface 
	"""

	print"\n--- Running Intrinsic Density and Curvature Routine --- \n"

	surf_dir = directory + '/surface'
	pos_dir = directory + '/intpos'
	den_dir = directory + '/intden'
	if not os.path.exists(den_dir): os.mkdir(den_dir)

	lslice = dim[2] / nslice

	file_name_pos = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
	file_name_coeff = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
	file_name_count = '{}_{}_{}_{}_{}_{}'.format(file_name, nslice, qm, n0, int(1./phi + 0.5), nframe)	
	file_name_norm = '{}_{}_{}_{}_{}_{}'.format(file_name, nz, qm, n0, int(1./phi + 0.5), nframe)

	if recon:
		file_name_pos += '_R'
		file_name_count += '{}_R'
		file_name_norm += '{}_R'

	if not os.path.exists('{}/intden/{}_count_corr.hdf5'.format(directory, file_name_count)):
		ut.make_hdf5(den_dir, file_name_count + '_count_corr', (qm+1, nslice, nz), tables.Float64Atom())
		ut.make_hdf5(den_dir, file_name_norm + '_z_nz', (qm+1, nz, nnz), tables.Float64Atom())
		file_check = False

	elif not ow_count:
		"Checking number of frames in current distribution files"
		try:
			file_check = (ut.shape_check_hdf5(den_dir, file_name_count + '_count_corr') == (nframe, 2, qm+1, nmol))
			file_check *= (ut.shape_check_hdf5(den_dir, file_name_norm + '_z_nz') == (nframe, 2, qm+1, nmol))
		except: file_check = False
	else:file_check = False

	if not file_check:
		_, _, zmol = ut.read_mol_positions(directory, file_name, nframe, nframe)
		COM = ut.read_com_positions(directory, file_name, nframe, nframe)
		nmol = zmol.shape[1]
		com_tile = np.moveaxis(np.tile(COM, (nmol, 1, 1)), [0, 1, 2], [2, 1, 0])[2]
		zmol = zmol - com_tile

		for frame in xrange(nframe):

			"Checking number of frames in hdf5 files"
			frame_check_count_corr = (ut.shape_check_hdf5(den_dir, file_name_count + '_count_corr')[0] <= frame)
			frame_check_z_nz = (ut.shape_check_hdf5(den_dir, file_name_norm + '_z_nz')[0] <= frame)

			if frame_check_count_corr: mode_count_corr = 'a'
			elif ow_count: mode_count_corr = 'r+'
			else: mode_count_corr = False

			if frame_check_z_nz: mode_z_nz = 'a'
			elif ow_count: mode_z_nz = 'r+'
			else: mode_z_nz = False

			if not mode_count_corr and not mode_z_nz:pass
			else:
				sys.stdout.write("Calculating position and curvature distributions: frame {}\r".format(frame))
				sys.stdout.flush()

				coeff = ut.load_hdf5(surf_dir, file_name_coeff + '_coeff', frame)
				int_z_mol = ut.load_hdf5(pos_dir, file_name_pos + '_int_z_mol', frame)
				int_dxdy_mol = ut.load_hdf5(pos_dir, file_name_pos + '_int_dxdy_mol', frame)

				count_corr_array, z_nz_array = make_den_curve(directory, zmol[frame], int_z_mol, int_dxdy_mol, coeff, nmol, nslice, nz, nnz, qm, dim)
				ut.save_hdf5(den_dir, file_name_count + '_count_corr', count_corr_array, frame, mode_count_corr)
				ut.save_hdf5(den_dir, file_name_norm + '_z_nz', z_nz_array, frame, mode_z_nz)


def intrinsic_density_dist(directory, file_name, nslice, qm, n0, phi, nframe, sample):

	den_dir = directory +'/intden'
	file_name_count = '{}_{}_{}_{}_{}_{}'.format(file_name, nslice, qm, n0, int(1./phi + 0.5), nframe)

	count_corr_array = ut.load_hdf5(den_dir, file_name_count + '_count_corr', 'all')

	count = np.rollaxis(count_corr_array, 0, 3)
	int_density = np.sum(count, axis=(2, 3))

	import matplotlib.pyplot as plt

	for dist in int_density:
		plt.plot(dist)
	plt.show()




