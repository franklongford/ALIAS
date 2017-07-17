"""
*************** INTRINSIC SURFACE MODULE *******************

Defines coefficients for a fouier series that represents
the periodic surfaces in the xy plane of an air-liquid 
interface. 	

***************************************************************
Created 24/11/16 by Frank Longford

Last modified 29/11/16 by Frank Longford
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

#from mpi4py import MPI

def intrinsic_surface(directory, model, csize, suffix, nsite, nmol, ncube, DIM, COM, nm, vlim, mol_sigma, M, image, nimage, ow_coeff):
	"Creates intrinsic surface of image." 

	phi = 5E-2
	if model.upper() == 'ARGON': c = 0.8
	else: c = 1.1

	max_r = 1.5 * mol_sigma
	tau = 0.4 * mol_sigma
	n0 = int(DIM[0] * DIM[1] * c / mol_sigma**2)

	if not os.path.exists("{}/DATA/ACOEFF".format(directory)): os.mkdir("{}/DATA/ACOEFF".format(directory))

	if os.path.exists('{}/DATA/ACOEFF/{}_{}_{}_{}_PIVOTS.txt'.format(directory, model.lower(), csize, nm, image)) and not ow_coeff:
	   	with file('{}/DATA/ACOEFF/{}_{}_{}_{}_INTCOEFF.txt'.format(directory, model.lower(), csize, nm, image), 'r') as infile: 
			auv1, auv2 = np.loadtxt(infile)
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PIVOTS.txt'.format(directory, model.lower(), csize, nm, image), 'r') as infile:
			piv_n1, piv_n2 = np.loadtxt(infile)
	else:
		sys.stdout.write("PROCESSING {} INTRINSIC SURFACE {} \n".format(directory, image) )
		sys.stdout.flush()

		xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, image)
		xR, yR, zR = COM[image]

		auv1, auv2, piv_n1, piv_n2 = build_surface(xmol, ymol, zmol, DIM, nmol, ncube, mol_sigma, nm, n0, vlim, phi, zR, tau, max_r)
	
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_INTCOEFF.txt'.format(directory, model.lower(), csize, nm, image), 'w') as outfile:
			np.savetxt(outfile, (auv1, auv2), fmt='%-12.6f')

		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PIVOTS.txt'.format(directory, model.lower(), csize, nm, image), 'w') as outfile:
			np.savetxt(outfile, (piv_n1, piv_n2), fmt='%-12.6f')

	return auv1, auv2, piv_n1, piv_n2


def build_surface(xmol, ymol, zmol, DIM, nmol, ncube, sigma, nm, n0, vlim, phi, zcom, tau, max_r):

	mol_list = range(nmol)
	piv_n1 = range(ncube**2)
	piv_z1 = np.zeros(ncube**2)
	piv_n2 = range(ncube**2)
	piv_z2 = np.zeros(ncube**2)
	new_pivots1 = []
	new_pivots2 = []

	for n in xrange(nmol):
		vapour = 0
		for m in xrange(nmol):
			dr2 = (xmol[n] - xmol[m])**2 + (ymol[n] - ymol[m])**2 + (zmol[n] - zmol[m])**2			
			if n!= m and dr2 < max_r**2: vapour += 1
			if vapour > vlim:

				indexx = int(xmol[n] * ncube / DIM[0]) % ncube
                                indexy = int(ymol[n] * ncube / DIM[1]) % ncube

				if zmol[n] - zcom < piv_z1[ncube*indexx + indexy]:
					piv_n1[ncube*indexx + indexy] = n
					piv_z1[ncube*indexx + indexy] = zmol[n] - zcom

				elif zmol[n] - zcom > piv_z2[ncube*indexx + indexy]:
					piv_n2[ncube*indexx + indexy] = n
					piv_z2[ncube*indexx + indexy] = zmol[n] - zcom

				break
		if vapour <= vlim: mol_list.remove(n)

	print np.array(piv_n1), np.array(piv_n2)
	for n in piv_n1: 
		mol_list.remove(n)
		new_pivots1.append(n)
	for n in piv_n2:
		mol_list.remove(n)
		new_pivots2.append(n)

	lpiv1 = len(new_pivots1)
	lpiv2 = len(new_pivots2)

	start = time.time()

	diag = np.diagflat([4*np.pi**2*((int(j/(2*nm+1))-nm)**2 + (int(j%(2*nm+1))-nm)**2)* phi for j in xrange((2*nm+1)**2)])
	A1 = np.zeros(((2*nm+1)**2, (2*nm+1)**2)) + diag
	A2 = np.zeros(((2*nm+1)**2, (2*nm+1)**2)) + diag
	b1 = np.zeros((2*nm+1)**2)
	b2 = np.zeros((2*nm+1)**2)

	loop = 0
	while len(piv_n1) < n0 or len(piv_n2) < n0 and lpiv1 + lpiv2 > 0:

		start1 = time.time()

		if lpiv1 > 0: 
			fu1 = [[function(xmol[ns], int(j/(2*nm+1))-nm, DIM[0]) * function(ymol[ns], int(j%(2*nm+1))-nm, DIM[1]) for ns in new_pivots1] for j in xrange((2*nm+1)**2)]
			for k in xrange((2*nm+1)**2): b1[k] += np.sum([(zmol[new_pivots1[ns]]- zcom) * fu1[k][ns] for ns in xrange(len(new_pivots1))])
		if lpiv2 > 0: 
			fu2 = [[function(xmol[ns], int(j/(2*nm+1))-nm, DIM[0]) * function(ymol[ns], int(j%(2*nm+1))-nm, DIM[1]) for ns in new_pivots2] for j in xrange((2*nm+1)**2)]
			for k in xrange((2*nm+1)**2): b2[k] += np.sum([(zmol[new_pivots2[ns]]- zcom) * fu2[k][ns] for ns in xrange(len(new_pivots2))])

		end11 = time.time()

		for j in xrange((2*nm+1)**2):
			for k in xrange(j+1):
				if lpiv1 > 0:
					A1[j][k] += np.sum([fu1[k][ns] * fu1[j][ns] for ns in xrange(len(new_pivots1))])
					A1[k][j] = A1[j][k]
				if lpiv2 > 0:
					A2[j][k] += np.sum([fu2[k][ns] * fu2[j][ns] for ns in xrange(len(new_pivots2))])
					A2[k][j] = A2[j][k]

		end1 = time.time()

		if lpiv1 > 0:
			lu, piv  = sp.linalg.lu_factor(A1)
			auv1 = sp.linalg.lu_solve((lu, piv), b1)
		if lpiv2 > 0:
			lu, piv  = sp.linalg.lu_factor(A2)
			auv2 = sp.linalg.lu_solve((lu, piv), b2)

		end2 = time.time()

		if len(piv_n1) == n0 and len(piv_n2) == n0: 
			print 'Matrix calculation: {:7.3f} {:7.3f}  Decomposition: {:7.3f} {} {} {} {} {} '.format(end11 - start1, end1 - end11, end2 - end1, n0, len(piv_n1), len(piv_n2), lpiv1, lpiv2)
			break
	
		new_pivots1 = []
		new_pivots2 = []

		for n in mol_list:

			x = xmol[n]
			y = ymol[n]
			z = zmol[n] - zcom

			if z < 0:
				zeta = xi(x, y, nm, auv1, DIM)
				if len(piv_n1) < n0 and abs(zeta - z) <= tau:
					piv_n1.append(n)
					new_pivots1.append(n)
					mol_list.remove(n)
				elif abs(zeta - z) > 3.0 * sigma:
					mol_list.remove(n)					
			else:
				zeta = xi(x, y, nm, auv2, DIM)
				if len(piv_n2) < n0 and abs(zeta - z) <= tau:
					piv_n2.append(n)
					new_pivots2.append(n)
					mol_list.remove(n)
				elif abs(zeta - z) > 3.0 * sigma:
					mol_list.remove(n)
			if len(piv_n1) == n0 and len(piv_n2) == n0: break

		end3 = time.time()

		lpiv1 = len(new_pivots1)
		lpiv2 = len(new_pivots2)

		tau = tau *1.1
		loop += 1

		end = time.time()
		print 'Matrix calculation: {:7.3f} {:7.3f}  Decomposition: {:7.3f}  Pivot selection: {:7.3f}  LOOP time: {:7.3f}   {} {} {} {} {} '.format(end11 - start1, end1 - end11, end2 - end1, end3 - end2, end - start1, n0, len(piv_n1), len(piv_n2), lpiv1, lpiv2)			

	print 'TOTAL time: {:7.2f}  {} {}'.format(end - start, len(piv_n1), len(piv_n2))
	return auv1, auv2, piv_n1, piv_n2


def function(x, u, Lx):

	if u == 0: return 1
	elif u > 0: return np.cos(2 * np.pi * u * x / Lx)
	else: return np.sin(- 2 * np.pi * u * x / Lx)


def dfunction(x, u, Lx):

	if u == 0: return 0
	elif u > 0: return - 2 * np.pi * u / Lx * np.sin(2 * np.pi * u * x / Lx)
	else: return -2 * np.pi * u / Lx * np.cos(-2 * np.pi * u * x / Lx)


def ddfunction(x, u, Lx):

	if u == 0: return 0
	elif u > 0: return - 4 * np.pi**2 * u**2 / Lx**2 * np.cos(2 * np.pi * u * x / Lx)
	else: return - 4 * np.pi**2 * u**2 / Lx**2 * np.sin(-2 * np.pi * u * x / Lx)

def xi(x, y, nm, auv, DIM):

	zeta = 0
	for u in xrange(-nm,nm+1):
		for v in xrange(-nm, nm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)
			zeta += function(x, u, DIM[0]) * function(y, v, DIM[1]) * auv[j]
	return zeta


def dxyi(x, y, nm, auv, DIM):

	dzx = 0
	dzy = 0
	for u in xrange(-nm,nm+1):
		for v in xrange(-nm, nm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)
			dzx += dfunction(x, u, DIM[0]) * function(y, v, DIM[1]) * auv[j]
			dzy += function(x, u, DIM[0]) * dfunction(y, v, DIM[1]) * auv[j]
	return dzx, dzy


def ddxyi(x, y, nm, auv, DIM):

	ddzx = 0
	ddzy = 0
	for u in xrange(-nm,nm+1):
		for v in xrange(-nm, nm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)
			ddzx += ddfunction(x, u, DIM[0]) * function(y, v, DIM[1]) * auv[j]
			ddzy += function(x, u, DIM[0]) * ddfunction(y, v, DIM[1]) * auv[j]
	return ddzx, ddzy


def slice_area(auv, nm, z):

        Axi = 0
        for j in xrange((2*nm+1)**2):
                xi2 = np.real(auv[j]*np.conj(auv[j]))
                Axi += xi2 / (1 + xi2 * abs(z) * j**2)**2
        return 1 + 0.5*Axi


def intrinsic_density(directory, COM, model, csize, suffix, nm, image, nslice, nsite, AT, DIM, M, auv1, auv2, ow_count):
	"Saves atom, mol and mass intrinsic profiles nimage number of trajectory snapshots" 
	
	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	if os.path.exists('{}/DATA/INTDEN/{}_{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), csize, nslice, nm, image)) and not ow_count:
		try:
			with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), csize, nslice, nm, image)) as infile:
				count_array = np.loadtxt(infile)
			for i in xrange(3 + n_atom_types): count_array[i]
			return count_array
		except IndexError: print "IndexError: len(count_array) != 2 + n_atom_types ({} {})".format(len(count_array), 2 + n_atom_types)

	make_intz = True

	if os.path.exists('{}/DATA/INTPOS/{}_{}_{}_{}_INTZ_AT.txt'.format(directory, model.lower(), csize, nm, image)):
		try:
			with file('{}/DATA/INTPOS/{}_{}_{}_{}_INTZ_AT.txt'.format(directory, model.lower(), csize, nm, image), 'r') as infile:
				int_z_at_1, int_z_at_2 = np.loadtxt(infile)
			with file('{}/DATA/INTPOS/{}_{}_{}_{}_INTZ_MOL.txt'.format(directory, model.lower(), csize, nm, image), 'r') as infile:
				int_z_mol_1, int_z_mol_2 = np.loadtxt(infile)
			make_intz = False
		except: make_intz = True
	if make_intz:
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_INTCOEFF.txt'.format(directory, model.lower(), csize, nm, image), 'r') as infile:
                        auv1, auv2 = np.loadtxt(infile)
		int_z_at_1 = []
		int_z_at_2 = []
		int_z_mol_1 = []
		int_z_mol_2 = []

	xat, yat, zat = ut.read_atom_positions(directory, model, csize, image)
	xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, image)
	xR, yR, zR = COM[image]

	count_array = [np.zeros(nslice) for n in range(3 + n_atom_types)]

	Aslice = DIM[0] * DIM[1]
	natom = len(xat)
	nmol = len(xmol)

       	for n in xrange(natom):
		sys.stdout.write("PROCESSING {} INTRINSIC DENSITY {}: make_intz = {}  {} out of {}  atoms\r".format(directory, image, make_intz, n, natom) )
		sys.stdout.flush()

		m = n % nsite
		at_type = AT[m]
	
		x = xat[n]
		y = yat[n]
		z = zat[n] - zR

		if make_intz: 
			int_z1 = xi(x, y, nm, auv1, DIM)
			int_z2 = xi(x, y, nm, auv2, DIM)
			int_z_at_1.append(int_z1)
			int_z_at_2.append(int_z2)
		else:
			int_z1 = int_z_at_1[n]
			int_z2 = int_z_at_2[n]
 
		z1 = z - int_z1
		z2 = -z + int_z2

		index1_at = int((z1 + DIM[2]/2.) * nslice / (DIM[2])) % nslice
		index2_at = int((z2 + DIM[2]/2.) * nslice / (DIM[2])) % nslice

		count_array[0][index1_at] += M[m]
		count_array[0][index2_at] += M[m]
		count_array[1 + atom_types.index(at_type)][index1_at] += 1./2
		count_array[1 + atom_types.index(at_type)][index2_at] += 1./2

		if m == 0:
			x = xmol[n/nsite]
			y = ymol[n/nsite]
			z = zmol[n/nsite] - zR

			if make_intz: 
				int_z1 = xi(x, y, nm, auv1, DIM)
				int_z2 = xi(x, y, nm, auv2, DIM)
				int_z_mol_1.append(int_z1)
				int_z_mol_2.append(int_z2)
			else:
				int_z1 = int_z_mol_1[n/nsite]
				int_z2 = int_z_mol_2[n/nsite]

			z1 = z - int_z1
			z2 = -z + int_z2

			index1_mol = int((z1 + DIM[2]/2.) * nslice / (DIM[2])) % nslice
			index2_mol = int((z2 + DIM[2]/2.) * nslice / (DIM[2])) % nslice

			#if index2_mol == 0: print  z, int_z2, z2, DIM[2]/2, int((z2 + DIM[2]/2.) * nslice / (DIM[2])), "\n"

			count_array[-2][index1_mol] += 1
			count_array[-1][index2_mol] += 1

	count_array[-1] = count_array[-1][::-1]

	with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), csize, nslice, nm, image), 'w') as outfile:
		np.savetxt(outfile, (count_array), fmt='%-12.6f')

	if make_intz: 
		with file('{}/DATA/INTPOS/{}_{}_{}_{}_INTZ_AT.txt'.format(directory, model.lower(), csize, nm, image), 'w') as outfile:
			np.savetxt(outfile, (int_z_at_1, int_z_at_2))

		with file('{}/DATA/INTPOS/{}_{}_{}_{}_INTZ_MOL.txt'.format(directory, model.lower(), csize, nm, image), 'w') as outfile:
			np.savetxt(outfile, (int_z_mol_1, int_z_mol_2))

	return count_array


def curve_mesh(directory, model, csize, nm, nxy, image, auv1, auv2, DIM, ow_curve):

	if not os.path.exists('{}/DATA/INTDEN/{}_{}_{}_{}_{}_CURVE.npz'.format(directory, model.lower(), csize, nm, nxy, image)) and not ow_curve:
		sys.stdout.write("PROCESSING {} SURFACE CURVE MESH {}\r".format(directory,  image) )
		sys.stdout.flush()

		X = np.linspace(0, DIM[0], nxy)
		Y = np.linspace(0, DIM[1], nxy)

		XI1 = np.zeros((nxy, nxy))
		XI2 = np.zeros((nxy, nxy))
		DX1 = np.zeros((nxy, nxy))
		DY1 = np.zeros((nxy, nxy))
		DX2 = np.zeros((nxy, nxy))
		DY2 = np.zeros((nxy, nxy))
		DDX1 = np.zeros((nxy, nxy))
		DDY1 = np.zeros((nxy, nxy))
		DDX2 = np.zeros((nxy, nxy))
		DDY2 = np.zeros((nxy, nxy))
		DXDY1 = np.zeros((nxy, nxy))
		DXDY2 = np.zeros((nxy, nxy))

		for j in xrange(nxy):
			x = X[j]
			for k in xrange(nxy):
				y = Y[k]
				for u in xrange(-nm,nm+1):
					for v in xrange(-nm, nm+1):
						l = (2 * nm + 1) * (u + nm) + (v + nm)
						XI1[j][k] += function(x, u, DIM[0]) * function(y, v, DIM[1]) * auv1[l]
						XI2[j][k] += function(x, u, DIM[0]) * function(y, v, DIM[1]) * auv2[l]

						DX1[j][k] += dfunction(x, u, DIM[0]) * function(y, v, DIM[1]) * auv1[l]
						DY1[j][k] += function(x, u, DIM[0]) * dfunction(y, v, DIM[1]) * auv1[l]
						DX2[j][k] += dfunction(x, u, DIM[0]) * function(y, v, DIM[1]) * auv2[l]
						DY2[j][k] += function(x, u, DIM[0]) * dfunction(y, v, DIM[1]) * auv2[l]

						DDX1[j][k] += ddfunction(x, u, DIM[0]) * function(y, v, DIM[1]) * auv1[l]
						DDY1[j][k] += function(x, u, DIM[0]) * ddfunction(y, v, DIM[1]) * auv1[l]
						DDX2[j][k] += ddfunction(x, u, DIM[0]) * function(y, v, DIM[1]) * auv2[l]
						DDY2[j][k] += function(x, u, DIM[0]) * ddfunction(y, v, DIM[1]) * auv2[l]

						DXDY1[j][k] += dfunction(x, u, DIM[0]) * dfunction(y, v, DIM[1]) * auv1[l]
						DXDY2[j][k] += dfunction(x, u, DIM[0]) * dfunction(y, v, DIM[1]) * auv2[l]

		with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_CURVE.npz'.format(directory, model.lower(), csize, nm, nxy, image), 'w') as outfile:
			np.savez(outfile, XI1=XI1, XI2=XI2, DX1=DX1, DY1=DY1, DX2=DX2, DY2=DY2, DDX1=DDX1, DDY1=DDY1, DDX2=DDX2, DDY2=DDY2, DXDY1=DXDY1, DXDY2=DXDY2)


def effective_density(directory, model, csize, nslice, nimage, suffix, DIM, nm, av_density_array, ow_cwden):

	print "\nBUILDING SLAB DENSITY PLOT {}/DATA/INTDEN/ CWDEN.txt".format(directory)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice)
	lslice = DIM[2] / nslice

	av_cw_den_1 = np.zeros(nslice)
        av_cw_den_2 = np.zeros(nslice)

	mean_auv1 = np.zeros(nimage)
	mean_auv2 = np.zeros(nimage)

	av_auv1_2 = np.zeros((2*nm+1)**2)
	av_auv2_2 = np.zeros((2*nm+1)**2)

	for image in xrange(nimage):
		sys.stdout.write("PROCESSING {} out of {} IMAGES\r".format(image, nimage) )
		sys.stdout.flush()

		with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), csize, nslice, nm, image)) as infile:
			count_array = np.loadtxt(infile)
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_INTCOEFF.txt'.format(directory, model.lower(), csize, nm, image), 'r') as infile:
			auv1, auv2 = np.loadtxt(infile)		

		mean_auv1[image] += auv1[len(auv1)/2]
                mean_auv2[image] += auv2[len(auv2)/2]

		auv1_2 = auv1**2
		auv2_2 = auv2**2

                av_auv1_2 += auv1_2 / nimage
                av_auv2_2 += auv2_2 / nimage

		"""
		"NUMERICAL EQUIVALENT"
		den_grid1 = np.zeros((nxy, nxy, nslice))
		den_grid2 = np.zeros((nxy, nxy, nslice))

		for j in xrange(nxy):
			for k in xrange(nxy):

				indent1 = nslice / 2 - int((XI1[j][k] + DIM[2]/2) / DIM[2] * nslice)
				indent2 = nslice / 2 - int((XI2[j][k] + DIM[2]/2) / DIM[2] * nslice)
				
				wave_den1 = np.array(list(mol_den1[indent1:]) + list(mol_den1[:indent1]))
				wave_den2 = np.array(list(mol_den2[indent2:]) + list(mol_den2[:indent2]))

				den_grid1[j][k] += wave_den1
				den_grid2[j][k] += wave_den2

		w_den_1_temp = np.array([np.mean(np.rollaxis(den_grid1, 2)[n]) for n in range(nslice)])
		w_den_2_temp = np.array([np.mean(np.rollaxis(den_grid2, 2)[n]) for n in range(nslice)])
		"""
		"""
		"CAPILLARY WAVE SMOOTHING PER SNAPSHOT"
		if not os.path.exists('{}/DATA/INTDEN/{}_{}_{}_{}_{}_CWDEN.txt'.format(directory, model.lower(), csize, nslice, nm, image)) or ow_cwden:
	
			cw_den_1 = np.zeros(nslice)
			cw_den_2 = np.zeros(nslice)
		
			mol_den1 = count_array[-2] * nslice / np.sum(np.array(DIM)**2)
			mol_den2 = count_array[-1] * nslice / np.sum(np.array(DIM)**2)

			Delta1 = (ut.sum_auv_2(auv1_2, nm) - mean_auv1[image]**2)
                	Delta2 = (ut.sum_auv_2(auv2_2, nm) - mean_auv2[image]**2)

                	P1_array = [ut.gaussian(z, 0, Delta1) for z in Z2]
                	P2_array = [ut.gaussian(z, 0, Delta2) for z in Z2]

			for n1, z1 in enumerate(Z1):
				for n2, z2 in enumerate(Z2):
					index1 = int((z1 - z2 - mean_auv1[image]) / DIM[2] * nslice) % nslice
					index2 = int((z1 - z2 - mean_auv2[image]) / DIM[2] * nslice) % nslice

					try: cw_den_1[n1] += mol_den1[index1] * P1_array[n2] * lslice
					except IndexError: pass

					try: cw_den_2[n1] += mol_den2[index2] * P2_array[n2] * lslice
					except IndexError: pass

			with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_CWDEN.txt'.format(directory, model.lower(), csize , nslice, nm, image), 'w') as outfile:
				np.savetxt(outfile, (cw_den_1, cw_den_2), fmt='%-12.6f')
		"""
	print ""

	"""
	with file('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, nimage), 'r') as infile:
		av_density = np.loadtxt(infile)
	rho = 0.5 * (av_cw_den_1 + av_cw_den_2[::-1])

	popt, pcov = curve_fit(gaussian_smoothing, rho, av_density[-1], p0=[1., np.mean(mean_auv1), DIM, nslice], bounds=([-np.inf, np.mean(mean_auv1), DIM, nslice], [np.inf, np.mean(mean_auv1), DIM, nslice])
	"""
	Delta1 = (ut.sum_auv_2(av_auv1_2, nm) - np.mean(mean_auv1)**2)
	Delta2 = (ut.sum_auv_2(av_auv2_2, nm) - np.mean(mean_auv2)**2)

	P1_array = [ut.gaussian(z, 0, Delta1) for z in Z2]
        P2_array = [ut.gaussian(z, 0, Delta2) for z in Z2]

	for n1, z1 in enumerate(Z1):
		for n2, z2 in enumerate(Z2):
			sys.stdout.write("PERFORMING GAUSSIAN SMOOTHING {0:.1%} COMPLETE\r".format(float(n1 * nslice + n2) / nslice**2) )
			sys.stdout.flush()

			index1 = int((z1 - z2 - np.mean(mean_auv1)) / DIM[2] * nslice) % nslice
			index2 = int((z1 - z2 - np.mean(mean_auv2)) / DIM[2] * nslice) % nslice

			try: av_cw_den_1[n1] += av_density_array[-4][index1] * P1_array[n2] * lslice
			except IndexError: pass

			try: av_cw_den_2[n1] += av_density_array[-3][index2] * P2_array[n2] * lslice
			except IndexError: pass		

	return av_cw_den_1, av_cw_den_2



def intrinsic_profile(directory, model, csize, suffix, nimage, natom, nmol, nsite, AT, M, mol_sigma, COM, DIM, nslice, ncube, nm, vlim, ow_coeff, ow_curve, ow_count, ow_wden):

	lslice = DIM[2] / nslice
	Aslice = DIM[0]*DIM[1]
	Vslice = DIM[0]*DIM[1]*lslice
	Acm = 1E-8

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	av_density_array = [np.zeros(nslice) for n in range(5 + n_atom_types)]

	start_image_count = 0
	start_image_curve = 0

	for image in xrange(nimage):
		auv1, auv2, piv_n1, piv_n2 = intrinsic_surface(directory, model, csize, suffix, nsite, nmol, ncube, DIM, COM, nm, vlim, mol_sigma, M, image, nimage, ow_coeff)
		#curve_mesh(directory, model, csize, nm, nxy, image, auv1, auv2, DIM, ow_curve)
		count_array = intrinsic_density(directory, COM, model, csize, suffix, nm, image, nslice, nsite, AT, DIM, M, auv1, auv2, ow_count)
		for i in xrange(3 + n_atom_types): av_density_array[i] += count_array[i] / (nimage * Vslice)

	av_density_array[0] = av_density_array[0] / (con.N_A * Acm**3)

	with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, nm, nimage), 'w') as outfile:
		np.savetxt(outfile, (av_density_array), fmt='%-12.6f')

	cw_den_1, cw_den_2 = effective_density(directory, model, csize, nslice, nimage, suffix, DIM, nm, av_density_array, ow_wden)

	av_density_array[-2] += cw_den_1
	av_density_array[-1] += cw_den_2

	print '\n'
	print "WRITING TO FILE..."

	with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, nm, nimage), 'w') as outfile:
		np.savetxt(outfile, (av_density_array), fmt='%-12.6f')

	print "{} {} {} COMPLETE\n".format(directory, model.upper(), csize)


def main(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, TYPE, folder, sfolder, nfolder, suffix, nimage=0):

	for i in xrange(sfolder, nfolder):
		if TYPE.upper() != 'SLAB': directory = '{}/{}_{}'.format(root, TYPE.upper(), i)
		else: directory = root
		traj = ut.load_nc(directory, folder, model, csize, suffix)						
		directory = '{}/{}'.format(directory, folder.upper())

		natom = traj.n_atoms
		nmol = traj.n_residues
		if nimage == 0: ntraj = traj.n_frames
		else: ntraj = nimage
		DIM = np.array(traj.unitcell_lengths[0]) * 10
		sigma = np.max(LJ[1])
		lslice = 0.05 * sigma
		nslice = int(DIM[2] / lslice)
		vlim = 3
		ncube = 3
		nm = int((DIM[0] + DIM[1]) / (2 * sigma))
		nxy = int((DIM[0]+DIM[1])/ sigma)

		if not os.path.exists("{}/DATA/INTDEN".format(directory)): os.mkdir("{}/DATA/INTDEN".format(directory))

		if os.path.exists('{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, nm, ntraj)):
			with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, nm, ntraj), 'r') as infile:
				check = np.loadtxt(infile)
			if np.sum(check[-1]) != 0:
				print "FILE FOUND '{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt".format(directory, model.lower(), csize, nslice, nm, ntraj)
				overwrite = raw_input("OVERWRITE? (Y/N): ")
				if overwrite.upper() == 'Y': 
					ow_coeff = raw_input("OVERWRITE ACOEFF? (Y/N): ")
					ow_curve = raw_input("OVERWRITE CURVE? (Y/N): ")
					ow_count = raw_input("OVERWRITE COUNT? (Y/N): ")
					ow_wden = raw_input("OVERWRITE WDEN? (Y/N): ")
					intrinsic_profile(traj, directory, csize, suffix, AT, DIM, M, ntraj, model, nsite, natom, nmol, sigma, nslice, ncube, nm, nxy, vlim, ow_coeff, ow_curve, ow_count, ow_wden)
			else: intrinsic_profile(traj, directory, csize, suffix, AT, DIM, M, ntraj, model, nsite, natom, nmol, sigma, nslice, ncube, nm, nxy, vlim, 'N','N', 'N', 'N')
		else: intrinsic_profile(traj, directory, csize, suffix, AT, DIM, M, ntraj, model, nsite, natom, nmol, sigma, nslice, ncube, nm, nxy, vlim, 'N','N','N', 'N')

