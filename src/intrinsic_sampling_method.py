"""
*************** INTRINSIC SAMPLING METHOD MODULE *******************

Defines coefficients for a fouier series that represents
the periodic surfaces in the xy plane of an air-liquid 
interface. 	

********************************************************************
Created 24/11/16 by Frank Longford

Last modified 06/02/18 by Frank Longford
"""

import numpy as np
import scipy as sp

import utilities as ut

import os, sys, time, tables

sqrt_2 = np.sqrt(2.)
vcheck = np.vectorize(ut.check_uv)
		
def make_coeff_pivots(directory, file_name, xmol, ymol, zmol, dim, mol_sigma, qm, n0, phi, frame, ncube=3, vlim=3, ow_coeff=False):
	"""
	make_coeff_pivots(directory, file_name, xmol, ymol, zmol, dim, mol_sigma, qm, n0, phi, psi, frame, ncube=3, vlim=3, recon=True, ow_coeff=False, ow_recon=False)

	Creates intrinsic surface of trajectory frame using molecular positions xmol, ymol, zmol.

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed.
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
		Maximum number of molecular pivots in intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	frame:  int
		Trajectory frame to analyse
	ncube:	int (optional)
		Grid size for initial pivot molecule selection
	vlim:  int (optional)
		Minimum number of molecular meighbours within radius max_r required for molecular NOT to be considered in vapour region
	ow_coeff:  bool (optional)
		Overwrite surface coefficients


	Returns
	-------

	coeff:	array_like (float); shape=(2, n_waves**2)
		Optimised surface coefficients
	pivots:  array_like (int); shape=(2, n0)
		Indicies of pivot molecules in molecular position arrays

	""" 

	n_waves = 2*qm+1
	max_r = 1.5 * mol_sigma
	tau = 0.5 * mol_sigma

	"Checking number of frames in coeff files"
	with tables.open_file('{}/surface/{}_coeff.hdf5'.format(directory, file_name), 'r') as infile:
		max_frame = infile.root.tot_coeff.shape[0]

	if max_frame <= frame and not ow_coeff: mode = 'a'
	elif ow_coeff: mode = 'r+'
	else: mode = False

	if not mode:
		coeff = load_coeff(directory, file_name, frame=frame)
		pivots = load_pivots(directory, file_name, frame=frame)

	else:
		sys.stdout.write("Optimising Intrinsic Surface coefficients: frame {}\n".format(frame))
		sys.stdout.flush()

		coeff, pivots = build_surface(xmol, ymol, zmol, dim, mol_sigma, qm, n0, phi, ncube, vlim, tau, max_r)
		save_coeff(directory, file_name, coeff, frame, n_waves, mode)
		save_pivots(directory, file_name, pivots, frame, n0, mode)

	return coeff, pivots


def make_recon_coeff(directory, file_name, coeff, pivots, xmol, ymol, zmol, dim, mol_sigma, qm, n0, phi, psi, frame, ow_coeff=False):
	"""
	make_recon_coeff(directory, file_name, coeff, pivots, xmol, ymol, zmol, dim, mol_sigma, qm, n0, phi, psi, frame, ow_coeff=False)

	Creates intrinsic surface of trajectory frame using molecular positions xmol, ymol, zmol.

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed.
	coeff:	array_like (float); shape=(2, n_waves**2)
		Optimised surface coefficients
	pivots:  array_like (int); shape=(2, n0)
		Indicies of pivot molecules in molecular position arrays
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
		Maximum number of molecular pivots in intrinsic surface
	phi:  float
		Weighting factor of minimum surface area term in surface optimisation function
	psi:  float
		Initial value for weighting factor of surface curvature variance in surface reconstruction function
	frame:  int
		Trajectory frame to analyse
	ow_coeff:  bool (optional)
		Overwrite surface coefficients	

	""" 

	n_waves = 2*qm+1 

	"Checking number of frames in reconstructed coeff files"
	with tables.open_file('{}/surface/{}_R_coeff.hdf5'.format(directory, file_name), 'r') as infile:
		max_frame = infile.root.tot_coeff.shape[0]

	if max_frame <= frame and not ow_coeff: mode = 'a'
	elif ow_coeff: mode = 'r+'
	else: mode = False

	if not mode:
		coeff_R = load_coeff(directory, file_name + '_R', frame=frame)
	else:
		sys.stdout.write("Reconstructing Intrinsic Surface coefficients: frame {}\r".format(frame))
		sys.stdout.flush()

		coeff_R = surface_reconstruction(coeff, pivots, xmol, ymol, zmol, dim, qm, n0, phi, psi)
		save_coeff(directory, file_name + '_R', coeff_R, frame, n_waves, mode)

	return coeff_R


def save_coeff(directory, file_name, coeff, frame, n_waves, mode='a'):
	"""
	save_coeff(directory, file_name, coeff, frame, n_waves, write='a')

	Save surface coefficients from frame in coeff.hdf5 file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	coeff:	array_like (float); shape=(2, n_waves**2)
		Optimised surface coefficients
	frame:  int
		Trajectory frame to save
	n_waves:  int
		Number of coefficients / waves in surface
	mode:  str (optional)
		Option to append 'a' to hdf5 file or overwrite 'rw' existing data	
	"""

	with tables.open_file('{}/surface/{}_coeff.hdf5'.format(directory, file_name), mode) as outfile:
		if mode.lower() == 'a':
			write_coeff = np.zeros((1, 2, n_waves**2))
			write_coeff[0] = coeff
			outfile.root.tot_coeff.append(write_coeff)
		elif mode.lower() == 'r+':
			outfile.root.tot_coeff[frame] = coeff


def save_pivots(directory, file_name, pivots, frame, n0, mode='a'):
	"""
	save_pivots(directory, file_name, pivots, frame, n0, mode='a')

	Save surface pivot molecules indicies from frame in pivot.hdf5 file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	pivots:  array_like (int); shape=(2, n0)
		Indicies of pivot molecules in molecular position arrays
	frame:  int
		Trajectory frame to save
	n0:  int
		Maximum number of molecular pivots in intrinsic surface
	mode:  str (optional)
		Option to append 'a' to hdf5 file or overwrite 'rw' existing data
	"""

	with tables.open_file('{}/surface/{}_pivot.hdf5'.format(directory, file_name), mode) as outfile:
		if mode.lower() == 'a':
			write_pivots = np.zeros((1, 2, n0))
			write_pivots[0] = pivots
			outfile.root.tot_pivot.append(write_pivots)
		elif mode.lower() == 'r+':
			outfile.root.tot_pivot[frame] = pivots


def load_coeff(directory, file_name, frame):
	"""
	load_coeff(directory, file_name, frame)

	Load surface coefficients from coeff.hdf5 file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	frame:  int (optional)
		Trajectory frame to load

	Returns
	-------

	coeff:	array_like (float); shape=(2, n_waves**2)
		Optimised surface coefficients
	"""

	with tables.open_file('{}/surface/{}_coeff.hdf5'.format(directory, file_name), 'r') as infile:
		if frame == 'all': coeff = infile.root.tot_coeff[:]
		else: coeff = infile.root.tot_coeff[frame]
	return coeff


def load_pivots(directory, file_name, frame):
	"""
	load_pivots(directory, file_name, frame)

	Load surface pivot molecular indicies from pivot.hdf5 file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	frame:  int (optional)
		Trajectory frame to load

	Returns
	-------

	pivots:  array_like (int); shape=(2, n0)
		Indicies of pivot molecules in molecular position arrays
	"""

	with tables.open_file('{}/surface/{}_pivot.hdf5'.format(directory, file_name), 'r') as infile:
		if frame == 'all': pivots = infile.root.tot_pivot[:]
		else: pivots = infile.root.tot_pivot[frame]
	return pivots


def numpy_remove(list1, list2):
	"""
	numpy_remove(list1, list2)

	Deletes overlapping elements of list2 from list1
	"""

	return np.delete(list1, np.where(np.isin(list1, list2)))


def build_surface(xmol, ymol, zmol, dim, mol_sigma, qm, n0, phi, ncube, vlim, tau, max_r):
					
	"""
	build_surface(xmol, ymol, zmol, dim, nmol, ncube, mol_sigma, qm, n0, phi, vlim, tau, max_r)

	Create coefficients auv1 and auv2 for Fourier sum representing intrinsic surface.

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
		Maximum number of molecular pivots in intrinsic surface
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
	pivots:  array_like (int); shape=(2, n0)
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

	dr2 = np.array((mat_xmol - np.transpose(mat_xmol))**2 + (mat_ymol - np.transpose(mat_ymol))**2 + (mat_zmol - np.transpose(mat_zmol))**2, dtype=float)
	
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
	mol_list = numpy_remove(mol_list, vapour_list)
	mol_list = numpy_remove(mol_list, piv_n1)
	mol_list = numpy_remove(mol_list, piv_n2)

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

	while building_surface:

		start1 = time.time()

		"Update A matrix and b vector"
		temp_A, temp_b = update_A_b(xmol, ymol, zmol, qm, n_waves, new_piv1, new_piv2, dim)

		A += temp_A
		b += temp_b

		end1 = time.time()

		"Perform LU decomosition to solve Ax = b"
		if len(new_piv1) != 0: auv1 = LU_decomposition(A[0] + diag, b[0])
		if len(new_piv2) != 0: auv2 = LU_decomposition(A[1] + diag, b[1])

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
		if build_surf1: zeta_list1 = zeta_list(xmol, ymol, mol_list1, auv1, qm, dim)
		if build_surf2: zeta_list2 = zeta_list(xmol, ymol, mol_list2, auv2, qm, dim)

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
		area1 = slice_area(auv1**2, qm, qm, dim)
		area2 = slice_area(auv2**2, qm, qm, dim)

		print ' {:20.3f}  {:20.3f}  {:20.3f}  {:10.3f} | {:10d} {:10d} {:10d} {:10d} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f}'.format(end1 - start1, end2 - end1, end - end2, end - start1, len(piv_n1), len(new_piv1), len(piv_n2), len(new_piv2), tau1, tau2, area1, area2)			

	print '\nTOTAL time: {:7.2f} s \n'.format(end - start)

	coeff = np.array((auv1, auv2))
	pivots = np.array((piv_n1, piv_n2), dtype=int)

	return coeff, pivots


def zeta_list(xmol, ymol, mol_list, auv, qm, dim):
	"""
	zeta_list(xmol, ymol, mol_list, auv, qm, dim)

	Calculate dz (zeta) between molecular sites an intrinsic surface for highest resolution"
	"""

	zeta_list = chi(xmol[mol_list], ymol[mol_list], qm, qm, auv, dim)
   
	return zeta_list


def pivot_selection(zmol, mol_sigma, n0, mol_list, zeta_list, piv_n, tau):
	"""
	pivot_selection(zmol, mol_sigma, n0, mol_list, zeta_list, piv_n, tau)

	Search through zeta_list for values within tau threshold and add to pivot list
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
	
	"Remove pivots form molecular search list"
	far_piv = mol_list[zeta > 6.0*tau]
	if len(new_piv) > 0: mol_list = numpy_remove(mol_list, np.append(new_piv, far_piv))
	
	assert np.sum(np.isin(new_piv, mol_list)) == 0

	return mol_list, new_piv, piv_n


def LU_decomposition(A, b):
	"""
	LU_decomposition(A, b)

	Perform lower-upper decomposition to solve equation Ax = b
	"""
	lu, piv  = sp.linalg.lu_factor(A)
	auv = sp.linalg.lu_solve((lu, piv), b)
	return auv


def update_A_b(xmol, ymol, zmol, qm, n_waves, new_piv1, new_piv2, dim):
	"""
	update_A_b(xmol, ymol, zmol, qm, n_waves, new_piv1, new_piv2, dim)
	
	Update A matrix and b vector for new pivot selection
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

	A[0] += np.dot(fuv1, np.transpose(fuv1))
	A[1] += np.dot(fuv2, np.transpose(fuv2))

	return A, b


def surface_reconstruction(coeff, pivots, xmol, ymol, zmol, dim, qm, n0, phi, psi):
	"""
	surface_reconstruction( xmol, ymol, zmol, dim, qm, n0, phi, psi)

	Reconstruct surface coefficients in Fourier sum representing intrinsic surface to yield expected variance of mean curvature.

	Parameters
	----------

	coeff:	array_like (float); shape=(2, n_waves**2)
		Optimised surface coefficients
	pivots:  array_like (int); shape=(2, n0)
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

	"Form the diagonal chi^2 terms"

	fuv1 = np.zeros((n_waves**2, n0))
        fuv2 = np.zeros((n_waves**2, n0))
	b = np.zeros((2, n_waves**2))

        for j in xrange(n_waves**2):
                fuv1[j] = wave_function(xmol[pivots[0]], int(j/n_waves)-qm, dim[0]) * wave_function(ymol[pivots[0]], int(j%n_waves)-qm, dim[1])
                b[0][j] += np.sum(zmol[pivots[0]] * fuv1[j])
                fuv2[j] = wave_function(xmol[pivots[1]], int(j/n_waves)-qm, dim[0]) * wave_function(ymol[pivots[1]], int(j%n_waves)-qm, dim[1])
                b[1][j] += np.sum(zmol[pivots[1]] * fuv2[j])

	u1 = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v1 = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

	diag = vcheck(u1, v1) * (phi * (u1**2 * dim[1] / dim[0] + v1**2 * dim[0] / dim[1]))
	diag = 4 * np.pi**2 * np.diagflat(diag)

	u1_tile = np.tile(u1, (n_waves**2, 1))
	v1_tile = np.tile(v1, (n_waves**2, 1))

	u2_tile = np.transpose(u1_tile)
	v2_tile = np.transpose(v1_tile)

	curve_weight = 16 * np.pi**4 * (u1_tile**2 * u2_tile**2 / dim[0]**4 + v1_tile**2 * v2_tile**2 / dim[1]**4 + (u1_tile**2 * v2_tile**2 + u2_tile**2 * v1_tile**2) / (dim[0]**2 * dim[1]**2))

	ffuv1 = np.dot(fuv1, np.transpose(fuv1))
	ffuv2 = np.dot(fuv2, np.transpose(fuv2))

	end_setup1 = time.time()

	print "{:^74s} | {:^21s} | {:^43s}".format('TIMINGS (s)', 'PSI', 'VAR(H)' )
	print ' {:20s} {:20s} {:20s} {:10s} | {:10s} {:10s} | {:10s} {:10s} {:10s} {:10s}'.format('Matrix Formation', 'LU Decomposition', 'Var Estimation', 'TOTAL', 'surf1', 'surf2', 'surf1', 'piv1', 'surf2', 'piv2')
	print "_" * 165

	H_var1 = ut.H_var_est(coeff[0]**2, qm, qm, dim)
	H_var2 = ut.H_var_est(coeff[1]**2, qm, qm, dim)

	auv1_matrix = np.tile(coeff[0], (n_waves**2, 1))
	H_piv_var1 = np.sum(auv1_matrix * np.transpose(auv1_matrix) * ffuv1 * curve_weight / n0)
	auv2_matrix = np.tile(coeff[1], (n_waves**2, 1))
	H_piv_var2 = np.sum(auv2_matrix * np.transpose(auv2_matrix) * ffuv2 * curve_weight / n0)

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
		A1 = ffuv1 * (1. + curve_weight * psi1 / n0)
		A2 = ffuv2 * (1. + curve_weight * psi2 / n0) 

		end1 = time.time()

		"Perform LU decomosition to solve Ax = b"
		if recon_1: auv1_recon = LU_decomposition(A1 + diag, b[0])
		if recon_2: auv2_recon = LU_decomposition(A2 + diag, b[1])

		end2 = time.time()

		H_var1_recon = ut.H_var_est(auv1_recon**2, qm, qm, dim)
		H_var2_recon = ut.H_var_est(auv2_recon**2, qm, qm, dim)

		if recon_1:
			auv1_matrix = np.tile(auv1_recon, (n_waves**2, 1))
			H_piv_var1_recon = np.sum(auv1_matrix * np.transpose(auv1_matrix) * ffuv1 * curve_weight / n0)
		if recon_2:
			auv2_matrix = np.tile(auv2_recon, (n_waves**2, 1))
			H_piv_var2_recon = np.sum(auv2_matrix * np.transpose(auv2_matrix) * ffuv2 * curve_weight / n0)

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

	coeff_recon = np.array((auv1_recon, auv2_recon))

	return coeff_recon


def wave_function(x, u, Lx):
	"""
	function(x, u, Lx)

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


def chi(x, y, qm, qu, auv, dim):
	"""
	chi(x, y, qm, qu, auv, dim)

	Function returning position of intrinsic surface at position (x,y)
	
	"""
	chi_z = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			chi_z += wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * auv[j]
	return chi_z


def dxy_dchi(x, y, qm, qu, auv, dim):
	"""
	dxy_dchi(x, y, qm, qu, auv, dim)

	Function returning derivatives of intrinsic surface at position (x,y) wrt x and y
	
	"""
	dx_dchi = 0
	dy_dchi = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			dx_dchi += d_wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * auv[j]
			dy_dchi += wave_function(x, u, dim[0]) * d_wave_function(y, v, dim[1]) * auv[j]
	return dx_dchi, dy_dchi


def ddxy_ddchi(x, y, qm, qu, auv, dim):
	"""
	ddxy_ddchi(x, y, qm, qu, auv, dim)

	Function returning second derivatives of intrinsic surface at position (x,y) wrt x and y
	
	"""
	ddx_ddchi = 0
	ddy_ddchi = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			ddx_ddchi += dd_wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * auv[j]
			ddy_ddchi += wave_function(x, u, dim[0]) * dd_wave_function(y, v, dim[1]) * auv[j]
	return ddx_ddchi, ddy_ddchi


def mean_curve_est(x, y, qm, qu, auv, dim):
	"""
	mean_curve_est(x, y, qm, qu, auv, dim)

	Estimation of variance of mean curvature, based on approximated Gaussian distribution
	"""

	H = 0
	for u in xrange(-qu, qu+1):
		for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			H += -4 * np.pi**2 * (u**2 / dim[0]**2 + v**2 / dim[1]**2) * wave_function(x, u, dim[0]) * wave_function(y, v, dim[1]) * auv[j]
	return H


def optimise_ns(directory, file_name, nmol, nsite, nframe, qm, phi, dim, mol_sigma, start_ns, step_ns, nframe_ns = 20):
	"""
	optimise_ns(directory, file_name, nmol, nsite, nframe, qm, phi, vlim, ncube, dim, mol_sigma, start_ns, step_ns, nframe_ns = 20)

	Routine to find optimised pivot density coefficient ns and pivot number n0 based on lowest pivot diffusion rate
	"""

	mol_ex_1 = []
	mol_ex_2 = []
	NS = []

	n_waves = 2 * qm + 1
	max_r = 1.5 * mol_sigma
	tau = 0.5 * mol_sigma

	xmol, ymol, zmol = ut.read_mol_positions(directory, file_name, nframe, nframe_ns)
	COM = ut.read_com_positions(directory, file_name, nframe, nframe_ns)

	if nframe < nframe_ns: nframe_ns = nframe
	ns = start_ns
	optimising = True

	while optimising:

		NS.append(ns)
		n0 = int(dim[0] * dim[1] * ns / mol_sigma**2)

		tot_piv_n1 = np.zeros((nframe_ns, n0))
		tot_piv_n2 = np.zeros((nframe_ns, n0))

		file_name_auv = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)

		if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv)):
			ut.make_earray('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv), 
				['tot_coeff'], tables.Float64Atom(), [(0, 2, n_waves**2)])
			ut.make_earray('{}/surface/{}_pivot.hdf5'.format(directory, file_name_auv), 
				['tot_pivot'], tables.Int64Atom(), [(0, 2, n0)])

		for frame in xrange(nframe_ns):
			coeff, pivots = make_coeff_pivots(directory, file_name_auv, xmol[frame], ymol[frame], zmol[frame]-COM[frame][2], dim, mol_sigma, qm, n0, phi, frame, ow_coeff=True)
			tot_piv_n1[frame] += pivots[0]
			tot_piv_n2[frame] += pivots[1]

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
			file_name_auv = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
			os.remove('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv))
			os.remove('{}/surface/{}_pivot.hdf5'.format(directory, file_name_auv))

	return opt_ns, opt_n0


def mol_exchange(piv_1, piv_2, nframe, n0):
	"""
	mol_exchange(piv_1, piv_2, nframe, n0)

	Calculates average diffusion rate of surface pivot molecules between frames 
	"""
	n_1 = 0
	n_2 = 0

	for frame in xrange(nframe-1):

		n_1 += len(set(piv_1[frame]) - set(piv_1[frame+1]))
		n_2 += len(set(piv_2[frame]) - set(piv_2[frame+1]))

	return n_1 / (n0 * float(nframe-1) * 1000), n_2 / (n0 * float(nframe-1) * 1000)



def create_intrinsic_surfaces(directory, file_name, dim, qm, n0, phi, mol_sigma, nframe, recon=True, ow_coeff=False, ow_recon=False):

	file_name_auv = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	n_waves = 2 * qm + 1

	"Make coefficient and pivot files"
	if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv)):
		ut.make_earray('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv), 
			['tot_coeff'], tables.Float64Atom(), [(0, 2, n_waves**2)])
		ut.make_earray('{}/surface/{}_pivot.hdf5'.format(directory, file_name_auv), 
			['tot_pivot'], tables.Int64Atom(), [(0, 2, n0)])
		file_check = False
	elif not ow_coeff:
		"Checking number of frames in current coefficient files"
		try:
			tot_coeff = np.zeros((nframe, 2, n_waves**2))
			tot_coeff += load_coeff(directory, file_name_auv, 'all')
			file_check = True
		except: file_check = False
	else: file_check = False

	if recon:
		psi = phi * dim[0] * dim[1]
		"Make recon coefficient file"
		if not os.path.exists('{}/surface/{}_R_coeff.hdf5'.format(directory, file_name_auv)):
			ut.make_earray('{}/surface/{}_R_coeff.hdf5'.format(directory, file_name_auv), 
				['tot_coeff'], tables.Float64Atom(), [(0, 2, n_waves**2)])
			file_check = False
		elif not ow_recon:
			"Checking number of frames in current recon coefficient files"
			try:
				tot_coeff_recon = np.zeros((nframe, 2, n_waves**2))
				tot_coeff_recon += load_coeff(directory, file_name_auv + '_R')
				file_check = True
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
			coeff, pivots = make_coeff_pivots(directory, file_name_auv, xmol[frame], ymol[frame], zmol[frame], dim, mol_sigma, qm, n0, phi, frame, ow_coeff=ow_coeff)
			if recon: make_recon_coeff(directory, file_name_auv, coeff, pivots, xmol[frame], ymol[frame], zmol[frame], dim, mol_sigma, qm, n0, phi, psi, frame, ow_coeff=ow_recon)



def slice_area(auv_2, qm, qu, dim):
	"""
	slice_area(auv_2, qm, qu, dim)

	Calculate the intrinsic surface area from square of coefficients auv_2
	"""

        Axi = 0.0

	for u in xrange(-qu, qu+1):
                for v in xrange(-qu, qu+1):
			j = (2 * qm + 1) * (u + qm) + (v + qm)
			dot_prod = np.pi**2  * (u**2/dim[0]**2 + v**2/dim[1]**2)

			if dot_prod != 0:
				f_2 = ut.check_uv(u, v) * auv_2[j]
				Axi += f_2 * dot_prod

        return 1 + 0.5*Axi


def gamma_q_auv(auv_2, qm, qu, DIM, T, q2_set):
	"""
	gamma_q_auv(auv_2, qm, qu, DIM, T, q2_set)

	Calculate frequency dependent surface tension spectrum across surface wave coefficients
	"""

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

	return gamma_hist * coeff


def cw_gamma_1(q, gamma, kappa): return gamma + kappa * q**2


def cw_gamma_2(q, gamma, kappa0, l0): return gamma + q**2 * (kappa0 + l0 * np.log(q))


def cw_gamma_dft(q, gamma, kappa, eta0, eta1): return gamma + eta0 * q + kappa * q**2 + eta1 * q**3


def cw_gamma_sk(q, gamma, w0, r0, dp): return gamma + np.pi/32 * w0 * r0**6 * dp**2 * q**2 * (np.log(q * r0 / 2.) - (3./4 * 0.5772))


def load_pos_derivatives(directory, file_name, frame):
	"""
	load_pos_derivatives(directory, file_name, frame)

	Load intrinsic molecular positions, and 1st and 2nd derivatives wrt x and y from hdf5 files

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	frame:  int (optional)
		Trajectory frame to load

	Returns
	-------

	int_z_mol:  array_like (float); shape=(nframe, 2, qm+1, nmol)
		Molecular distances from intrinsic surface
	int_dxdy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
		First derivatives of intrinsic surface wrt x and y at xmol, ymol
	int_ddxddy_mol:  array_like (float); shape=(nframe, 4, qm+1, nmol)
		Second derivatives of intrinsic surface wrt x and y at xmol, ymol 
	"""

	with tables.open_file('{}/intpos/{}_int_z_mol.hdf5'.format(directory, file_name), 'r') as infile:
		if frame == 'all': int_z_mol = infile.root.int_z_mol[:]
		else: int_z_mol = infile.root.int_z_mol[frame]
	with tables.open_file('{}/intpos/{}_int_dxdy_mol.hdf5'.format(directory, file_name), 'r') as infile:
		if frame == 'all': int_z_mol = infile.root.int_dxdy_mol[:]
		else: int_dxdy_mol = infile.root.int_dxdy_mol[frame]
	with tables.open_file('{}/intpos/{}_int_ddxddy_mol.hdf5'.format(directory, file_name), 'r') as infile:
		if frame == 'all': int_z_mol = infile.root.int_ddxddy_mol[:]
		else: int_ddxddy_mol = infile.root.int_ddxddy_mol[frame]

	return int_z_mol, int_dxdy_mol, int_ddxddy_mol


def save_pos_derivatives(directory, file_name, int_z_mol, int_dxdy_mol, int_ddxddy_mol, frame, qm, nmol, mode='a'):
	"""
	save_coeff(directory, file_name, coeff, frame, n_waves, write='a')

	Save surface coefficients from frame in coeff.hdf5 file

	Parameters
	----------

	directory:  str
		File path of directory of alias analysis.
	file_name:  str
		File name of trajectory being analysed
	coeff:	array_like (float); shape=(2, n_waves**2)
		Optimised surface coefficients
	frame:  int
		Trajectory frame to save
	n_waves:  int
		Number of coefficients / waves in surface
	mode:  str (optional)
		Option to append 'a' to hdf5 file or overwrite 'rw' existing data	
	"""


	with tables.open_file('{}/intpos/{}_int_z_mol.hdf5'.format(directory, file_name), mode) as outfile:
		if mode.lower() == 'a':
			write_int_z_mol = np.zeros((1, 2, qm+1, nmol))
			write_int_z_mol[0] = int_z_mol
			outfile.root.int_z_mol.append(write_int_z_mol)
		elif mode.lower() == 'r+':
			outfile.root.int_z_mol[frame] = int_z_mol
	with tables.open_file('{}/intpos/{}_int_dxdy_mol.hdf5'.format(directory, file_name), mode) as outfile:
		if mode.lower() == 'a':
			write_dxdy_mol = np.zeros((1, 4, qm+1, nmol))
			write_dxdy_mol[0] = int_dxdy_mol
			outfile.root.int_dxdy_mol.append(write_dxdy_mol)
		elif mode.lower() == 'r+':
			outfile.root.int_dxdy_mol[frame] = int_dxdy_mol
	with tables.open_file('{}/intpos/{}_int_ddxddy_mol.hdf5'.format(directory, file_name), mode) as outfile:
		if mode.lower() == 'a':
			write_ddxddy_mol = np.zeros((1, 4, qm+1, nmol))
			write_ddxddy_mol[0] = int_ddxddy_mol
			outfile.root.tot_coeff.append(write_ddxddy_mol)
		elif mode.lower() == 'r+':
			outfile.root.int_ddxddy_mol[frame] = int_ddxddy_mol


def create_intrinsic_positions_dxdyz(directory, file_name, nmol, nframe, nsite, qm, n0, phi, dim, recon, ow_pos):
	"""
	intrinsic_positions_dxdyz(directory, file_name, xmol, ymol, zmol, auv1, auv2, frame, nframe, nsite, qm, n0, phi, psi, dim, recon, ow_pos)

	Calculate distances and derivatives at each atomic position with respect to intrinsic surface in simulation frame 
	"""

	n_waves = 2 * qm + 1

	file_name_auv = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	if recon: file_name_auv += '_R'

	file_name_pos = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	if recon: file_name_pos += '_R'

	if not os.path.exists('{}/intpos/{}_int_z_mol.hdf5'.format(directory, file_name_pos)):
		ut.make_earray('{}/intpos/{}_int_z_mol.hdf5'.format(directory, file_name_pos), 
			['int_z_mol'], tables.Float64Atom(), [(0, 2, qm+1, nmol)])
		ut.make_earray('{}/intpos/{}_int_dxdy_mol.hdf5'.format(directory, file_name_pos), 
			['int_dxdy_mol'], tables.Float64Atom(), [(0, 4, qm+1, nmol)])
		ut.make_earray('{}/intpos/{}_int_ddxddy_mol.hdf5'.format(directory, file_name_pos), 
			['int_ddxddy_mol'], tables.Float64Atom(), [(0, 4, qm+1, nmol)])
		file_check = False

	elif not ow_pos:
		"Checking number of frames in current distance files"
		try:
			tot_int_z_mol = np.zeros((2, qm+1, nframe, nmol))
			tot_int_z_mol += load_int_z_mol(directory, file_name_pos, 'all')
			file_check = True
		except: file_check = False
	else: file_check = False

	print file_check

	if not file_check:
		xmol, ymol, _ = ut.read_mol_positions(directory, file_name, nframe, nframe)
		tot_coeff = load_coeff(directory, file_name_auv, 'all')

		for frame in xrange(nframe):

			"Checking number of frames in int_z_mol file"
			with tables.open_file('{}/intpos/{}_int_z_mol.hdf5'.format(directory, file_name_pos), 'r') as infile:
				max_frame = infile.root.int_z_mol.shape[0]

			if max_frame <= frame and not ow_pos: mode = 'a'
			elif ow_pos: mode = 'r+'
			else: mode = False

			if not mode: pass
			else:
				sys.stdout.write("Calculating molecular distances and derivatives: frame {}\r".format(frame))
				sys.stdout.flush()

				int_z_mol, int_dxdy_mol, int_ddxddy_mol = make_pos_derivatives(directory, file_name_pos, xmol[frame], ymol[frame], tot_coeff[frame], frame, nmol, dim, qm)
				save_pos_derivatives(directory, file_name_pos, int_z_mol, int_dxdy_mol, int_ddxddy_mol, frame, qm, nmol, mode)


def make_pos_derivatives(directory, file_name_pos, xmol, ymol, coeff, frame, nmol, dim, qm):

	
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


def intrinsic_z_den_corr(directory, file_name, zmol, auv1, auv2, qm, n0, phi, psi, frame, nframe, nslice, nsite, nz, nnz, DIM, recon, ow_count):
	"""
	intrinsic_z_den_corr(directory, file_name, zmol, auv1, auv2, qm, n0, phi, psi, frame, nframe, nslice, nsite, nz, nnz, DIM, recon, ow_count)

	Calculate density and curvature distributions across surface 
	"""

	lslice = DIM[2] / nslice

	file_name_pos = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
	file_name_count = '{}_{}_{}_{}_{}_{}'.format(file_name, nslice, qm, n0, int(1./phi + 0.5), nframe)	
	file_name_norm = '{}_{}_{}_{}_{}_{}'.format(file_name, nz, qm, n0, int(1./phi + 0.5), nframe)

	if recon:
		file_name_pos = '{}_R'.format(file_name_pos)
		file_name_count = '{}_R'.format(file_name_count)
		file_name_norm = '{}_R'.format(file_name_norm)

	try:
		with tables.open_file('{}/intden/{}_countcorr.hdf5'.format(directory, file_name_count), 'r') as infile:
			count_corr_array = infile.root.count_corr_array[frame]
		with tables.open_file('{}/intden/{}_n_nz.hdf5'.format(directory, file_name_norm), 'r') as infile:
			z_nz_array = infile.root.z_nz_array[frame]
        except: ow_count = True

	if ow_count:

		write_count_corr_array = np.zeros((1, qm+1, nslice, nnz))
		write_z_nz_array = np.zeros((1, qm+1, nz, nnz))
		count_corr_array = np.zeros((qm+1, nslice, nnz))
		z_nz_array = np.zeros((qm+1, nz, nnz))	

		nmol = len(zmol)

		with tables.open_file('{}/intpos/{}_intz_mol.hdf5'.format(directory, file_name_pos), 'r') as infile:
			int_z_mol = infile.root.int_z_mol[frame]
		with tables.open_file('{}/intpos/{}_intdxdy_mol.hdf5'.format(directory, file_name_pos), 'r') as infile:
			dxdyz_mol = infile.root.dxdyz_mol[frame]

		for qu in xrange(qm+1):
			sys.stdout.write("PROCESSING {} INTRINSIC DENSITY {}: qm = {} qu = {}\r".format(directory, frame, qm, qu) )
			sys.stdout.flush()

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

			temp_count_corr_array += np.histogram2d(index1_mol, index1_nz, bins=[nslice, nnz], range=[[0, nslice], [0, nnz]])[0]
			temp_count_corr_array += np.histogram2d(index2_mol, index2_nz, bins=[nslice, nnz], range=[[0, nslice], [0, nnz]])[0]

			index1_mol = np.array(abs(int_z1 - auv1[len(auv1)/2]) * 2 * nz / (nz*lslice), dtype=int) % nz
			index2_mol = np.array(abs(int_z2 - auv2[len(auv2)/2]) * 2 * nz / (nz*lslice), dtype=int) % nz

			temp_z_nz_array += np.histogram2d(index1_mol, index1_nz, bins=[nz, nnz], range=[[0, nz], [0, nnz]])[0]
			temp_z_nz_array += np.histogram2d(index2_mol, index2_nz, bins=[nz, nnz], range=[[0, nz], [0, nnz]])[0]

			count_corr_array[qu] += temp_count_corr_array
			z_nz_array[qu] += temp_z_nz_array

		with tables.open_file('{}/intden/{}_countcorr.hdf5'.format(directory, file_name_count), 'a') as outfile:
			write_count_corr_array[0] = count_corr_array
			outfile.root.count_corr_array.append(write_count_corr_array)
		with tables.open_file('{}/intden/{}_n_nz.hdf5'.format(directory, file_name_norm), 'a') as outfile:
			write_z_nz_array[0] = z_nz_array
			outfile.root.z_nz_array.append(write_z_nz_array)

	return count_corr_array, z_nz_array



def intrinsic_profile(directory, file_name, T, nframe, natom, nmol, nsite, AT, M, mol_sigma, mol_com, dim, nslice, ncube, qm, QM, n0, phi, psi, npi, vlim, ow_coeff, ow_recon, ow_pos, ow_intden, ow_count, ow_effden):

	lslice = dim[2] / nslice
	Aslice = dim[0]*dim[1]
	Vslice = dim[0]*dim[1]*lslice
	Acm = 1E-8
	ur = 1
	Z1 = np.linspace(0, dim[2], nslice)
	Z2 = np.linspace(-dim[2]/2, dim[2]/2, nslice)
	NZ = np.linspace(0, 1, 100)
	n_waves = 2 * qm + 1
	nz = 100
        nnz = 100

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	av_auv1 = np.zeros((2, nframe))
	av_auv2 = np.zeros((2, nframe))

	av_auv1_2 = np.zeros((2, n_waves**2))
	av_auv2_2 = np.zeros((2, n_waves**2))

	av_auvU_2 = np.zeros((2, n_waves**2))
        av_auvP_2 = np.zeros((2, n_waves**2))

	tot_auv1 = np.zeros((2, nframe, n_waves**2))
	tot_auv2 = np.zeros((2, nframe, n_waves**2))

	file_name_pos = ['{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe),
			 '{}_{}_{}_{}_{}_R'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)]
	file_name_auv = ['{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe), 
			 '{}_{}_{}_{}_{}_R'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)]
	file_name_den = ['{}_{}_{}_{}_{}_{}'.format(file_name, nslice, qm, n0, int(1/phi + 0.5), nframe), 
		         '{}_{}_{}_{}_{}_{}_R'.format(file_name, nslice, qm, n0, int(1/phi + 0.5), nframe)]
	file_name_count = ['{}_{}_{}_{}_{}_{}'.format(file_name, nslice, qm, n0, int(1./phi + 0.5), nframe),
			   '{}_{}_{}_{}_{}_{}_R'.format(file_name, nslice, qm, n0, int(1./phi + 0.5), nframe)]	
	file_name_norm = ['{}_{}_{}_{}_{}_{}'.format(file_name, nz, qm, n0, int(1./phi + 0.5), nframe),
			  '{}_{}_{}_{}_{}_{}_R'.format(file_name, nz, qm, n0, int(1./phi + 0.5), nframe)]

	if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv[0])) or ow_coeff:
		ut.make_earray('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv[0]), 
			['tot_coeff1', 'tot_coeff2'], tables.Float64Atom(), [(0, n_waves**2), (0, n_waves**2)])
		ut.make_earray('{}/surface/{}_pivots.hdf5'.format(directory, file_name_auv[0]), 
			['tot_piv_n1', 'tot_piv_n2'], tables.Float64Atom(), [(0, n0), (0, n0)])
	if not os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv[1])) or ow_recon:
		ut.make_earray('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv[1]), 
			['tot_coeff1', 'tot_coeff2'], tables.Float64Atom(), [(0, n_waves**2), (0, n_waves**2)])

	for r, recon in enumerate([False, True]):
		if not os.path.exists('{}/intpos/{}_intz_mol.hdf5'.format(directory, file_name_pos[r])) or ow_pos:
			ut.make_earray('{}/intpos/{}_intz_mol.hdf5'.format(directory, file_name_pos[r]), 
				['int_z_mol'], tables.Float64Atom(), [(0, qm+1, 2, nmol)])
		if not os.path.exists('{}/intpos/{}_intdxdy_mol.hdf5'.format(directory, file_name_pos[r])) or ow_pos:
			ut.make_earray('{}/intpos/{}_intdxdy_mol.hdf5'.format(directory, file_name_pos[r]), 
				['dxdyz_mol'], tables.Float64Atom(), [(0, qm+1, 4, nmol)])
		if not os.path.exists('{}/intpos/{}_intddxddy_mol.hdf5'.format(directory, file_name_pos[r])) or ow_pos:
			ut.make_earray('{}/intpos/{}_intddxddy_mol.hdf5'.format(directory, file_name_pos[r]), 
				['ddxddyz_mol'], tables.Float64Atom(), [(0, qm+1, 4, nmol)])
		if not os.path.exists('{}/intden/{}_countcorr.hdf5'.format(directory, file_name_count[r])) or ow_count:
			ut.make_earray('{}/intden/{}_countcorr.hdf5'.format(directory, file_name_count[r]), 
				['count_corr_array'], tables.Float64Atom(), [(0, qm+1, nslice, 100)])
		if not os.path.exists('{}/intden/{}_n_nz.hdf5'.format(directory, file_name_norm[r])) or ow_count:
			ut.make_earray('{}/intden/{}_n_nz.hdf5'.format(directory, file_name_norm[r]), 
				['z_nz_array'], tables.Float64Atom(), [(0, qm+1, 100, 100)])

	file_check = [os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv[0])),
		      	os.path.exists('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv[1])),	
			os.path.exists('{}/surface/{}_pivots.hdf5'.format(directory, file_name_auv[0])),
			os.path.exists('{}/intpos/{}_intz_mol.hdf5'.format(directory, file_name_pos[0])),
			os.path.exists('{}/intpos/{}_intz_mol.hdf5'.format(directory, file_name_pos[1])),
			os.path.exists('{}/intpos/{}_intdxdy_mol.hdf5'.format(directory, file_name_pos[0])),
			os.path.exists('{}/intpos/{}_intdxdy_mol.hdf5'.format(directory, file_name_pos[1])),
			os.path.exists('{}/intpos/{}_intddxddy_mol.hdf5'.format(directory, file_name_pos[0])),
			os.path.exists('{}/intpos/{}_intddxddy_mol.hdf5'.format(directory, file_name_pos[1]))]

	file_check = np.all(file_check)

	if file_check:
		try:
			for r, recon in enumerate([False, True]):
				with tables.open_file('{}/surface/{}_coeff.hdf5'.format(directory, file_name_auv[r]), 'r') as infile:
					tot_auv1[r] = infile.root.tot_coeff1
					tot_auv2[r] = infile.root.tot_coeff2
		except: 
			tot_auv1 = np.zeros((2, nframe, n_waves**2))
			tot_auv2 = np.zeros((2, nframe, n_waves**2))
			file_check = False

	if not file_check or ow_coeff or ow_recon or ow_pos:

		print "IMPORTING GLOBAL POSITION DISTRIBUTIONS\n"

		xat, yat, zat = ut.read_atom_positions(directory, file_name, nframe, nframe)
		xmol, ymol, zmol = ut.read_mol_positions(directory, file_name, nframe, nframe)
		COM = ut.read_com_positions(directory, file_name, nframe, nframe)

		for frame in xrange(nframe):
			for r, recon in enumerate([False, True]):
				tot_auv1[r][frame], tot_auv2[r][frame], piv_n1, piv_n2 = intrinsic_surface(directory, file_name_auv[0], 
					xmol[frame], ymol[frame], zmol[frame]-COM[frame][2], dim, mol_sigma, qm, n0, phi, psi, frame,
					recon=recon, ow_coeff=ow_coeff, ow_recon=ow_recon)
				intrinsic_positions_dxdyz(directory, file_name, xmol[frame], ymol[frame], zmol[frame]-COM[frame][2], 
					tot_auv1[r][frame], tot_auv2[r][frame], frame, nframe, nsite, qm, n0, phi, psi, dim, recon, ow_pos)

				av_auv1_2[r] += tot_auv1[r][frame]**2 / nframe
				av_auv2_2[r] += tot_auv2[r][frame]**2 / nframe

				av_auv1[r][frame] = tot_auv1[r][frame][n_waves**2/2]
				av_auv2[r][frame] = tot_auv2[r][frame][n_waves**2/2]

				av_auvU_2[r] += (tot_auv1[r][frame] + tot_auv2[r][frame])**2/ (4. * nframe)
				av_auvP_2[r] += (tot_auv1[r][frame] - tot_auv2[r][frame])**2/ (4. * nframe)
		
		del xat, yat, zat, xmol, ymol, zmol, COM
		gc.collect()

	else:
		sys.stdout.write("LOADING INTRINSIC SURFACE PROFILES {} frames\n".format(nframe) )
		sys.stdout.flush()

		for r, recon in enumerate([False, True]):
		
			av_auv1_2[r] += np.sum(tot_auv1[r]**2, axis=0) / nframe
			av_auv2_2[r] += np.sum(tot_auv2[r]**2, axis=0) / nframe

			av_auv1[r] += np.swapaxes(tot_auv1, 1, 2)[r][n_waves**2/2]
			av_auv2[r] += np.swapaxes(tot_auv2, 1, 2)[r][n_waves**2/2]

			av_auvU_2[r] += np.sum(tot_auv1[r] + tot_auv2[r], axis=0)**2/ (4. * nframe)
			av_auvP_2[r] += np.sum(tot_auv1[r] - tot_auv2[r], axis=0)**2/ (4. * nframe)

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

	for r, recon in enumerate([False, True]):

		file_check = np.all([os.path.exists('{}/intden/{}_eff_den.npy'.format(directory, file_name_den[r]))])

		if not file_check or ow_effden:

			file_check = np.all([os.path.exists('{}/intden/{}_mol_den.npy'.format(directory, file_name_den[r])),
			     	     os.path.exists('{}/intden/{}_mol_den_corr.npy'.format(directory, file_name_den[r]))])

			if not file_check or ow_intden:

				xat, yat, zat = ut.read_atom_positions(directory, file_name, nframe, nframe)
				_, _, zmol = ut.read_mol_positions(directory, file_name, nframe, nframe)
				COM = ut.read_com_positions(directory, file_name, nframe, nframe)

				av_den_corr_matrix = np.zeros((qm+1, nslice, 100))
				av_z_nz_matrix = np.zeros((qm+1, 100, 100))	

				for frame in xrange(nframe):
					sys.stdout.write("PROCESSING INTRINSIC DISTRIBUTIONS {} out of {} frames\r".format(frame, nframe) )
					sys.stdout.flush() 

					int_count_corr_array, int_count_z_nz = intrinsic_z_den_corr(directory, file_name, zmol[frame]-COM[frame][2], tot_auv1[r][frame], tot_auv2[r][frame], qm, n0, phi, psi, frame, nframe, nslice, nsite, nz, nnz, dim, recon, ow_count)
					av_den_corr_matrix += int_count_corr_array
					av_z_nz_matrix += int_count_z_nz

				N = np.linspace(0, 50 * lslice, 100)

				int_den_corr = av_den_corr_matrix / (2 * nframe * Vslice)
				mol_int_den = np.sum(int_den_corr, axis=2)

				with file('{}/intden/{}_mol_den.npy'.format(directory, file_name_den[r]), 'w') as outfile:
					np.save(outfile, mol_int_den)
				with file('{}/intden/{}_mol_den_corr.npy'.format(directory, file_name_den[r]), 'w') as outfile:
					np.save(outfile, int_den_corr)

			else: mol_int_den = np.load('{}/intden/{}_mol_den.npy'.format(directory, file_name_den[r]), mmap_mode='r')

			eff_den = np.zeros((qm+1, nslice))

			for qu in xrange(1, qm+1):

				q_set = []
				q2_set = []

				for u in xrange(-qu, qu):
					for v in xrange(-qu, qu):
						q = 4 * np.pi**2 * (u**2 / dim[0]**2 + v**2/dim[1]**2)
						q2 = u**2 * dim[1]/dim[0] + v**2 * dim[0]/dim[1]

						if q2 not in q2_set:
							q_set.append(q)
							q2_set.append(np.round(q2, 4))

				q_set = np.sqrt(np.sort(q_set, axis=None))
				q2_set = np.sort(q2_set, axis=None)
				Q_set.append(q_set)

				AU[r][qu] = (slice_area(av_auvU_2[r], qm, qu, dim))
				AP[r][qu] = (slice_area(av_auvP_2[r], qm, qu, dim))
				ACU[r][qu] = (slice_area(av_auvCU_2[r], qm, qu, dim))

				cw_gammaU[r].append(gamma_q_auv(av_auvU_2[r]*2, qm, qu, dim, T, q2_set))
				cw_gammaP[r].append(gamma_q_auv(av_auvP_2[r]*2, qm, qu, dim, T, q2_set))
				cw_gammaCU[r].append(gamma_q_auv(av_auvCU_2[r], qm, qu, dim, T, q2_set))

				if qu == qm:
					file_name_gamma = ['{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe),
							'{}_{}_{}_{}_{}_R'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)]

					with file('{}/intthermo/{}_gamma.npy'.format(directory, file_name_gamma[r]), 'w') as outfile:
						np.save(outfile, (Q_set[-1], cw_gammaU[r][-1], cw_gammaP[r][-1], cw_gammaCU[r][-1]))

				Delta1 = (ut.sum_auv_2(av_auv1_2[r], qm, qu) - np.mean(av_auv1[r])**2)
				Delta2 = (ut.sum_auv_2(av_auv2_2[r], qm, qu) - np.mean(av_auv2[r])**2)

				centres = np.ones(2) * (np.mean(av_auv1[r]) - np.mean(av_auv2[r]))/2.
				deltas = np.ones(2) * 0.5 * (Delta1 + Delta2)

				arrays = [mol_int_den[qu], mol_int_den[qu]]

				cw_arrays = ut.gaussian_smoothing(arrays, centres, deltas, dim, nslice)

				eff_den[qu] = cw_arrays[0]

				print '\n'
				print "WRITING TO FILE... qm = {}  qu = {}  var1 = {}  var2 = {}".format(qm, qu, Delta1, Delta2)

			with file('{}/intden/{}_eff_den.npy'.format(directory, file_name_den[r]), 'w') as outfile:
				np.save(outfile, eff_den)
		else:
			with file('{}/intden/{}_eff_den.npy'.format(directory, file_name_den[r]), 'r') as infile:
				eff_den = np.load(infile)

	print "INTRINSIC SAMPLING METHOD {} {} {} {} {} COMPLETE\n".format(directory, file_name, qm, n0, phi)

	return av_auv1, av_auv2 
