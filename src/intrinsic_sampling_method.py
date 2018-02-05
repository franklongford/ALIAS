"""
*************** INTRINSIC SAMPLING METHOD MODULE *******************

Defines coefficients for a fouier series that represents
the periodic surfaces in the xy plane of an air-liquid 
interface. 	

********************************************************************
Created 24/11/16 by Frank Longford

Last modified 22/08/17 by Frank Longford
"""

sqrt_2 = np.sqrt(2.)
vcheck = np.vectorize(ut.check_uv)
		
def intrinsic_surface(directory, file_name, xmol, ymol, zmol, dim, nmol, ncube, qm, n0, phi, psi, vlim, mol_sigma, frame, nframe, recon, ow_auv, ow_recon):
	"Creates intrinsic surface of frame." 

	n_waves = 2*qm+1
	write_auv = np.zeros((1, n_waves**2))
	write_piv_n = np.zeros((1, n0))
	max_r = 1.5 * mol_sigma
	tau = 0.5 * mol_sigma

	with tables.open_file('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name), 'r') as infile:
		max_frame = infile.root.tot_coeff1.shape[0]

	sys.stdout.write("PROCESSING INTRINSIC SURFACE PROFILES {} out of {} frames\r".format(frame, nframe) )
	sys.stdout.flush()

	if max_frame <= frame or ow_auv:

		auv1, auv2, piv_n1, piv_n2 = build_surface(xmol, ymol, zmol, dim, nmol, ncube, mol_sigma, qm, n0, phi, vlim, tau, max_r)

		with tables.open_file('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name), 'a') as outfile:
			write_auv[0] = auv1
			outfile.root.tot_coeff1.append(write_auv)
			write_auv[0] = auv2
			outfile.root.tot_coeff2.append(write_auv)
		with tables.open_file('{}/SURFACE/{}_PIVOTS.hdf5'.format(directory, file_name), 'a') as outfile:
			write_piv_n[0] = piv_n1
			outfile.root.tot_piv_n1.append(write_piv_n)
			write_piv_n[0] = piv_n2
			outfile.root.tot_piv_n2.append(write_piv_n)
	else:
		with tables.open_file('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name), 'r') as infile:
			auv1 = infile.root.tot_coeff1[frame]
			auv2 = infile.root.tot_coeff2[frame]
		with tables.open_file('{}/SURFACE/{}_PIVOTS.hdf5'.format(directory, file_name), 'r') as infile:
			piv_n1 = infile.root.tot_piv_n1[frame]
			piv_n2 = infile.root.tot_piv_n2[frame]

	if recon:
		with tables.open_file('{}/SURFACE/{}_R_ACOEFF.hdf5'.format(directory, file_name), 'r') as infile:
			max_frame_R = infile.root.tot_coeff1.shape[0]

		if max_frame_R <= frame or ow_recon:
			sys.stdout.write("PROCESSING {}\nINTRINSIC SURFACE RECONSTRUCTION {}\n".format(directory, frame) )
			sys.stdout.flush()

			auv1_recon, auv2_recon = surface_reconstruction(xmol, ymol, zmol, qm, n0, phi, psi, auv1, auv2, np.array(piv_n1, dtype=int), np.array(piv_n2, dtype=int), dim)
		
			with tables.open_file('{}/SURFACE/{}_R_ACOEFF.hdf5'.format(directory, file_name), 'a') as outfile:
				write_auv[0] = auv1_recon
				outfile.root.tot_coeff1.append(write_auv)
				write_auv[0] = auv2_recon
				outfile.root.tot_coeff2.append(write_auv)
		else: 
			with tables.open_file('{}/SURFACE/{}_R_ACOEFF.hdf5'.format(directory, file_name), 'r') as infile:
				auv1_recon = infile.root.tot_coeff1[frame]
				auv2_recon = infile.root.tot_coeff2[frame]

		return auv1, auv2, auv1_recon, auv2_recon, np.array(piv_n1, dtype=int), np.array(piv_n2, dtype=int)

	else: return auv1, auv2, np.array(piv_n1, dtype=int), np.array(piv_n2, dtype=int)
	
		
def numpy_remove(list1, list2):

	return np.delete(list1, np.where(np.isin(list1, list2)))


def build_surface(xmol, ymol, zmol, DIM, nmol, ncube, mol_sigma, qm, n0, phi, vlim, tau, max_r):
	"Create coefficients auv1 and auv2 for Fourier sum representing intrinsic surface"

	print "\n ------------- BUILDING INTRINSIC SURFACE --------------"

	"""
	xmol, ymol, zmol = x, y, z positions of molecules
	mol_list = index list of molecules eligible to be used as 'pivots' for the fitting routine  
	piv_n = index of pivot molecules
	new_pivots = list of pivots to be added to piv_n and piv_z
	"""
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
	mol_list = numpy_remove(mol_list, vapour_list)#[i for i in mol_list if i not in vapour_list]
	mol_list = numpy_remove(mol_list, piv_n1)#[i for i in mol_list if i not in piv_n1]
	mol_list = numpy_remove(mol_list, piv_n2)#np.array([i for i in mol_list if i not in piv_n2])

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

	"Form the diagonal xi^2 terms"
	u = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
        v = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

	diag = vcheck(u, v) * (u**2 * DIM[1] / DIM[0] + v**2 * DIM[0] / DIM[1])
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
		temp_A, temp_b = update_A_b(xmol, ymol, zmol, qm, n_waves, new_piv1, new_piv2, DIM)

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
		if build_surf1: zeta_list1 = zeta_list(xmol, ymol, mol_list1, auv1, qm, DIM)
		if build_surf2: zeta_list2 = zeta_list(xmol, ymol, mol_list2, auv2, qm, DIM)

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
		area1 = slice_area(auv1**2, qm, qm, DIM)
		area2 = slice_area(auv2**2, qm, qm, DIM)

		print ' {:20.3f}  {:20.3f}  {:20.3f}  {:10.3f} | {:10d} {:10d} {:10d} {:10d} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f}'.format(end1 - start1, end2 - end1, end - end2, end - start1, len(piv_n1), len(new_piv1), len(piv_n2), len(new_piv2), tau1, tau2, area1, area2)			

	print '\nTOTAL time: {:7.2f} s \n'.format(end - start)

	return auv1, auv2, piv_n1, piv_n2


def zeta_list(xmol, ymol, mol_list, auv, qm, DIM):
	"Calculate dz (zeta) between molecular sites an intrinsic surface for highest resolution"

	zeta_list = xi(xmol[mol_list], ymol[mol_list], qm, qm, auv, DIM)
   
	return zeta_list


def pivot_selection(zmol, mol_sigma, n0, mol_list, zeta_list, piv_n, tau):
	"Search through zeta_list for values within tau threshold and add to pivot list"

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
	if len(new_piv) > 0: mol_list = numpy_remove(mol_list, np.append(new_piv, far_piv))#np.array([i for i in mol_list if i not in new_piv])
	#mol_list = numpy_remove(mol_list, new_piv)
	
	assert np.sum(np.isin(new_piv, mol_list)) == 0

	return mol_list, new_piv, piv_n


def LU_decomposition(A, b):
	"Perform lower-upper decomposition to solve equation Ax = b"
	lu, piv  = sp.linalg.lu_factor(A)
	auv = sp.linalg.lu_solve((lu, piv), b)
	return auv


def update_A_b(xmol, ymol, zmol, qm, n_waves, new_piv1, new_piv2, DIM):
	"Update A matrix and b vector for new pivot selection"

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

	fuv1 = np.zeros((n_waves**2, n0))
        fuv2 = np.zeros((n_waves**2, n0))
	b = np.zeros((2, n_waves**2))

        for j in xrange(n_waves**2):
                fuv1[j] = function(xmol[piv_n1], int(j/n_waves)-qm, DIM[0]) * function(ymol[piv_n1], int(j%n_waves)-qm, DIM[1])
                b[0][j] += np.sum(zmol[piv_n1] * fuv1[j])
                fuv2[j] = function(xmol[piv_n2], int(j/n_waves)-qm, DIM[0]) * function(ymol[piv_n2], int(j%n_waves)-qm, DIM[1])
                b[1][j] += np.sum(zmol[piv_n2] * fuv2[j])

	u1 = np.array(np.arange(n_waves**2) / n_waves, dtype=int) - qm
	v1 = np.array(np.arange(n_waves**2) % n_waves, dtype=int) - qm

	diag = vcheck(u1, v1) * (phi * (u1**2 * DIM[1] / DIM[0] + v1**2 * DIM[0] / DIM[1]))
	diag = 4 * np.pi**2 * np.diagflat(diag)

	u1_tile = np.tile(u1, (n_waves**2, 1))
	v1_tile = np.tile(v1, (n_waves**2, 1))

	u2_tile = np.transpose(u1_tile)
	v2_tile = np.transpose(v1_tile)

	coeff = 16 * np.pi**4 * (u1_tile**2 * u2_tile**2 / DIM[0]**4 + v1_tile**2 * v2_tile**2 / DIM[1]**4 + (u1_tile**2 * v2_tile**2 + u2_tile**2 * v1_tile**2) / (DIM[0]**2 * DIM[1]**2))

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
		if recon_1: auv1_recon = LU_decomposition(A1 + diag, b[0])
		if recon_2: auv2_recon = LU_decomposition(A2 + diag, b[1])

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


def optimise_ns(directory, file_name, nmol, nsite, nframe, qm, phi, vlim, ncube, dim, mol_sigma, start_ns, step_ns):

	mol_ex_1 = []
	mol_ex_2 = []
	NS = []

	n_waves = 2 * qm + 1
	max_r = 1.5 * mol_sigma
	tau = 0.5 * mol_sigma

	if not os.path.exists("{}/SURFACE".format(directory)): os.mkdir("{}/SURFACE".format(directory))

	nframe_ns = 20

	xmol, ymol, zmol = ut.read_mol_positions(directory, file_name, nframe, nframe_ns)
	COM = ut.read_com_positions(directory, file_name, nframe, nframe_ns)

	ns = start_ns
	optimising = True

	while optimising:

		NS.append(ns)
		n0 = int(dim[0] * dim[1] * ns / mol_sigma**2)

		tot_piv_n1 = np.zeros((nframe_ns, n0))
		tot_piv_n2 = np.zeros((nframe_ns, n0))

		file_name_auv = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)

		if not os.path.exists('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name_auv)):
			ut.make_earray('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name_auv), 
				['tot_coeff1', 'tot_coeff2'], tables.Float64Atom(), [(0, n_waves**2), (0, n_waves**2)])
			ut.make_earray('{}/SURFACE/{}_PIVOTS.hdf5'.format(directory, file_name_auv), 
				['tot_piv_n1', 'tot_piv_n2'], tables.Float64Atom(), [(0, n0), (0, n0)])

		for frame in xrange(nframe_ns):
			auv1, auv2, piv_n1, piv_n2 = intrinsic_surface(directory, file_name_auv, xmol[frame], ymol[frame], zmol[frame]-COM[frame][2], dim, nmol, ncube, qm, n0, phi, 0, vlim, mol_sigma, frame, nframe, False, False, False)
			tot_piv_n1[frame] += piv_n1
			tot_piv_n2[frame] += piv_n2

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
			os.remove('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name_auv))
			os.remove('{}/SURFACE/{}_PIVOTS.hdf5'.format(directory, file_name_auv))

	return opt_ns, opt_n0


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


def intrinsic_positions_dxdyz(directory, file_name, xmol, ymol, zmol, auv1, auv2, frame, nframe, nsite, qm, n0, phi, psi, dim, recon, ow_pos):

	file_name_pos = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1/phi + 0.5), nframe)
	if recon: file_name_pos = '{}_R'.format(file_name_pos)

	try:
		with tables.open_file('{}/INTPOS/{}_INTZ_MOL.hdf5'.format(directory, file_name_pos), 'r') as infile:
			int_z_mol = infile.root.int_z_mol[frame]
		with tables.open_file('{}/INTPOS/{}_INTDXDY_MOL.hdf5'.format(directory, file_name_pos), 'r') as infile:
			dxdyz_mol = infile.root.dxdyz_mol[frame]
		with tables.open_file('{}/INTPOS/{}_INTDDXDDY_MOL.hdf5'.format(directory, file_name_pos), 'r') as infile:
			ddxddyz_mol = infile.root.ddxddyz_mol[frame]
        except: ow_pos = True

	if ow_pos:

		nmol = len(xmol)

		write_int_z_mol = np.zeros((1, qm+1, 2, nmol))
		write_dxdyz_mol = np.zeros((1, qm+1, 4, nmol)) 
		write_ddxddyz_mol = np.zeros((1, qm+1, 4, nmol))

		int_z_mol = np.zeros((qm+1, 2, nmol))
		dxdyz_mol = np.zeros((qm+1, 4, nmol)) 
		ddxddyz_mol = np.zeros((qm+1, 4, nmol))

		temp_int_z_mol = np.zeros((2, nmol))
		temp_dxdyz_mol = np.zeros((4, nmol)) 
		temp_ddxddyz_mol = np.zeros((4, nmol))

		for qu in xrange(qm+1):
			sys.stdout.write("PROCESSING {} INTRINSIC POSITIONS AND DXDY {}: qm = {} qu = {}\r".format(directory, frame, qm, qu))
			sys.stdout.flush()

			if qu == 0:
				j = (2 * qm + 1) * qm + qm
				f_x = function(xmol, 0, dim[0])
				f_y = function(ymol, 0, dim[1])

				temp_int_z_mol[0] += f_x * f_y * auv1[j]
				temp_int_z_mol[1] += f_x * f_y * auv2[j]

			else:
				for u in [-qu, qu]:
					for v in xrange(-qu, qu+1):
						j = (2 * qm + 1) * (u + qm) + (v + qm)

						f_x = function(xmol, u, dim[0])
						f_y = function(ymol, v, dim[1])
						df_dx = dfunction(xmol, u, dim[0])
						df_dy = dfunction(ymol, v, dim[1])
						ddf_ddx = ddfunction(xmol, u, dim[0])
						ddf_ddy = ddfunction(ymol, v, dim[1])

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

						f_x = function(xmol, u, dim[0])
						f_y = function(ymol, v, dim[1])
						df_dx = dfunction(xmol, u, dim[0])
						df_dy = dfunction(ymol, v, dim[1])
						ddf_ddx = ddfunction(xmol, u, dim[0])
						ddf_ddy = ddfunction(ymol, v, dim[1])

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

		with tables.open_file('{}/INTPOS/{}_INTZ_MOL.hdf5'.format(directory, file_name_pos), 'a') as outfile:
			write_int_z_mol[0] = int_z_mol
			outfile.root.int_z_mol.append(write_int_z_mol)
		with tables.open_file('{}/INTPOS/{}_INTDXDY_MOL.hdf5'.format(directory, file_name_pos), 'a') as outfile:
			write_dxdyz_mol[0] = dxdyz_mol
			outfile.root.dxdyz_mol.append(write_dxdyz_mol)
		with tables.open_file('{}/INTPOS/{}_INTDDXDDY_MOL.hdf5'.format(directory, file_name_pos), 'a') as outfile:
			write_ddxddyz_mol[0] = ddxddyz_mol
			outfile.root.ddxddyz_mol.append(write_ddxddyz_mol)

	return int_z_mol, dxdyz_mol, ddxddyz_mol



def intrinsic_z_den_corr(directory, file_name, zmol, auv1, auv2, qm, n0, phi, psi, frame, nframe, nslice, nsite, nz, nnz, DIM, recon, ow_count):
	"Saves atom, mol and mass intrinsic profiles of trajectory frame" 

	lslice = DIM[2] / nslice

	file_name_pos = '{}_{}_{}_{}_{}'.format(file_name, qm, n0, int(1./phi + 0.5), nframe)
	file_name_count = '{}_{}_{}_{}_{}_{}'.format(file_name, nslice, qm, n0, int(1./phi + 0.5), nframe)	
	file_name_norm = '{}_{}_{}_{}_{}_{}'.format(file_name, nz, qm, n0, int(1./phi + 0.5), nframe)

	if recon:
		file_name_pos = '{}_R'.format(file_name_pos)
		file_name_count = '{}_R'.format(file_name_count)
		file_name_norm = '{}_R'.format(file_name_norm)

	try:
		with tables.open_file('{}/INTDEN/{}_COUNTCORR.hdf5'.format(directory, file_name_count), 'r') as infile:
			count_corr_array = infile.root.count_corr_array[frame]
		with tables.open_file('{}/INTDEN/{}_N_NZ.hdf5'.format(directory, file_name_norm), 'r') as infile:
			z_nz_array = infile.root.z_nz_array[frame]
        except: ow_count = True

	if ow_count:

		write_count_corr_array = np.zeros((1, qm+1, nslice, nnz))
		write_z_nz_array = np.zeros((1, qm+1, nz, nnz))
		count_corr_array = np.zeros((qm+1, nslice, nnz))
		z_nz_array = np.zeros((qm+1, nz, nnz))	

		nmol = len(zmol)

		with tables.open_file('{}/INTPOS/{}_INTZ_MOL.hdf5'.format(directory, file_name_pos), 'r') as infile:
			int_z_mol = infile.root.int_z_mol[frame]
		with tables.open_file('{}/INTPOS/{}_INTDXDY_MOL.hdf5'.format(directory, file_name_pos), 'r') as infile:
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

		with tables.open_file('{}/INTDEN/{}_COUNTCORR.hdf5'.format(directory, file_name_count), 'a') as outfile:
			write_count_corr_array[0] = count_corr_array
			outfile.root.count_corr_array.append(write_count_corr_array)
		with tables.open_file('{}/INTDEN/{}_N_NZ.hdf5'.format(directory, file_name_norm), 'a') as outfile:
			write_z_nz_array[0] = z_nz_array
			outfile.root.z_nz_array.append(write_z_nz_array)

	return count_corr_array, z_nz_array


def cw_gamma_1(q, gamma, kappa): return gamma + kappa * q**2


def cw_gamma_2(q, gamma, kappa0, l0): return gamma + q**2 * (kappa0 + l0 * np.log(q))


def cw_gamma_dft(q, gamma, kappa, eta0, eta1): return gamma + eta0 * q + kappa * q**2 + eta1 * q**3


def cw_gamma_sk(q, gamma, w0, r0, dp): return gamma + np.pi/32 * w0 * r0**6 * dp**2 * q**2 * (np.log(q * r0 / 2.) - (3./4 * 0.5772))


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

	if not os.path.exists('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name_auv[0])) or ow_coeff:
		ut.make_earray('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name_auv[0]), 
			['tot_coeff1', 'tot_coeff2'], tables.Float64Atom(), [(0, n_waves**2), (0, n_waves**2)])
		ut.make_earray('{}/SURFACE/{}_PIVOTS.hdf5'.format(directory, file_name_auv[0]), 
			['tot_piv_n1', 'tot_piv_n2'], tables.Float64Atom(), [(0, n0), (0, n0)])
	if not os.path.exists('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name_auv[1])) or ow_recon:
		ut.make_earray('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name_auv[1]), 
			['tot_coeff1', 'tot_coeff2'], tables.Float64Atom(), [(0, n_waves**2), (0, n_waves**2)])

	for r, recon in enumerate([False, True]):
		if not os.path.exists('{}/INTPOS/{}_INTZ_MOL.hdf5'.format(directory, file_name_pos[r])) or ow_pos:
			ut.make_earray('{}/INTPOS/{}_INTZ_MOL.hdf5'.format(directory, file_name_pos[r]), 
				['int_z_mol'], tables.Float64Atom(), [(0, qm+1, 2, nmol)])
		if not os.path.exists('{}/INTPOS/{}_INTDXDY_MOL.hdf5'.format(directory, file_name_pos[r])) or ow_pos:
			ut.make_earray('{}/INTPOS/{}_INTDXDY_MOL.hdf5'.format(directory, file_name_pos[r]), 
				['dxdyz_mol'], tables.Float64Atom(), [(0, qm+1, 4, nmol)])
		if not os.path.exists('{}/INTPOS/{}_INTDDXDDY_MOL.hdf5'.format(directory, file_name_pos[r])) or ow_pos:
			ut.make_earray('{}/INTPOS/{}_INTDDXDDY_MOL.hdf5'.format(directory, file_name_pos[r]), 
				['ddxddyz_mol'], tables.Float64Atom(), [(0, qm+1, 4, nmol)])
		if not os.path.exists('{}/INTDEN/{}_COUNTCORR.hdf5'.format(directory, file_name_count[r])) or ow_count:
			ut.make_earray('{}/INTDEN/{}_COUNTCORR.hdf5'.format(directory, file_name_count[r]), 
				['count_corr_array'], tables.Float64Atom(), [(0, qm+1, nslice, 100)])
		if not os.path.exists('{}/INTDEN/{}_N_NZ.hdf5'.format(directory, file_name_norm[r])) or ow_count:
			ut.make_earray('{}/INTDEN/{}_N_NZ.hdf5'.format(directory, file_name_norm[r]), 
				['z_nz_array'], tables.Float64Atom(), [(0, qm+1, 100, 100)])

	file_check = [os.path.exists('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name_auv[0])),
		      	os.path.exists('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name_auv[1])),	
			os.path.exists('{}/SURFACE/{}_PIVOTS.hdf5'.format(directory, file_name_auv[0])),
			os.path.exists('{}/INTPOS/{}_INTZ_MOL.hdf5'.format(directory, file_name_pos[0])),
			os.path.exists('{}/INTPOS/{}_INTZ_MOL.hdf5'.format(directory, file_name_pos[1])),
			os.path.exists('{}/INTPOS/{}_INTDXDY_MOL.hdf5'.format(directory, file_name_pos[0])),
			os.path.exists('{}/INTPOS/{}_INTDXDY_MOL.hdf5'.format(directory, file_name_pos[1])),
			os.path.exists('{}/INTPOS/{}_INTDDXDDY_MOL.hdf5'.format(directory, file_name_pos[0])),
			os.path.exists('{}/INTPOS/{}_INTDDXDDY_MOL.hdf5'.format(directory, file_name_pos[1]))]

	file_check = np.all(file_check)

	if file_check:
		try:
			for r, recon in enumerate([False, True]):
				with tables.open_file('{}/SURFACE/{}_ACOEFF.hdf5'.format(directory, file_name_auv[r]), 'r') as infile:
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
		
			tot_auv1[0][frame], tot_auv2[0][frame], tot_auv1[1][frame],tot_auv2[1][frame], piv_n1, piv_n2 = intrinsic_surface(directory, file_name_auv[0], xmol[frame], ymol[frame], zmol[frame]-COM[frame][2], dim, nmol, ncube, qm, n0, phi, psi, vlim, mol_sigma, frame, nframe, True, ow_coeff, ow_recon)

			for r, recon in enumerate([False, True]):
				intrinsic_positions_dxdyz(directory, file_name, xmol[frame], ymol[frame], zmol[frame]-COM[frame][2], tot_auv1[r][frame], tot_auv2[r][frame], frame, nframe, nsite, qm, n0, phi, psi, dim, recon, ow_pos)

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

		file_check = np.all([os.path.exists('{}/INTDEN/{}_EFF_DEN.npy'.format(directory, file_name_den[r]))])

		if not file_check or ow_effden:

			file_check = np.all([os.path.exists('{}/INTDEN/{}_MOL_DEN.npy'.format(directory, file_name_den[r])),
			     	     os.path.exists('{}/INTDEN/{}_MOL_DEN_CORR.npy'.format(directory, file_name_den[r]))])

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

				with file('{}/INTDEN/{}_MOL_DEN.npy'.format(directory, file_name_den[r]), 'w') as outfile:
					np.save(outfile, mol_int_den)
				with file('{}/INTDEN/{}_MOL_DEN_CORR.npy'.format(directory, file_name_den[r]), 'w') as outfile:
					np.save(outfile, int_den_corr)

			else: mol_int_den = np.load('{}/INTDEN/{}_MOL_DEN.npy'.format(directory, file_name_den[r]), mmap_mode='r')

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

					with file('{}/INTTHERMO/{}_GAMMA.npy'.format(directory, file_name_gamma[r]), 'w') as outfile:
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

			with file('{}/INTDEN/{}_EFF_DEN.npy'.format(directory, file_name_den[r]), 'w') as outfile:
				np.save(outfile, eff_den)

		else:
			with file('{}/INTDEN/{}_EFF_DEN.npy'.format(directory, file_name_den[r]), 'r') as infile:
				eff_den = np.load(infile)

	print "INTRINSIC SAMPLING METHOD {} {} {} {} {} COMPLETE\n".format(directory, file_name, qm, n0, phi)

	return av_auv1, av_auv2 
