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

def intrinsic_surface(directory, model, csize, nsite, nmol, ncube, DIM, COM, nm, n0, phi, vlim, mol_sigma, M, frame, nframe, ow_coeff):
	"Creates intrinsic surface of frame." 

	max_r = 1.5 * mol_sigma
	tau = 0.4 * mol_sigma

	if not os.path.exists("{}/DATA/ACOEFF".format(directory)): os.mkdir("{}/DATA/ACOEFF".format(directory))

	if os.path.exists('{}/DATA/ACOEFF/{}_{}_{}_{}_{}_PIVOTS.txt'.format(directory, model.lower(), nm, n0, int(1./phi + 0.5), frame)) and not ow_coeff:
	   	with file('{}/DATA/ACOEFF/{}_{}_{}_{}_{}_INTCOEFF.txt'.format(directory, model.lower(), nm, n0, int(1./phi + 0.5), frame), 'r') as infile: 
			auv1, auv2 = np.loadtxt(infile)
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_{}_PIVOTS.txt'.format(directory, model.lower(), nm, n0, int(1./phi + 0.5), frame), 'r') as infile:
			piv_n1, piv_n2 = np.loadtxt(infile)
	else:
		sys.stdout.write("PROCESSING {} INTRINSIC SURFACE {} {} {} {}\n".format(directory, frame, nm, n0, phi) )
		sys.stdout.flush()

		xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)
		xR, yR, zR = COM[frame]

		auv1, auv2, piv_n1, piv_n2 = build_surface(xmol, ymol, zmol, DIM, nmol, ncube, mol_sigma, nm, n0, phi, vlim, zR, tau, max_r)
	
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_{}_INTCOEFF.txt'.format(directory, model.lower(), nm, n0, int(1./phi + 0.5), frame), 'w') as outfile:
			np.savetxt(outfile, (auv1, auv2), fmt='%-12.6f')
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_{}_PIVOTS.txt'.format(directory, model.lower(), nm, n0, int(1./phi + 0.5), frame), 'w') as outfile:
			np.savetxt(outfile, (piv_n1, piv_n2))

		intrinsic_positions(directory, model, csize, frame, auv1, auv2, natom, nmol, nsite, nm, nm, n0, phi, DIM, ow_coeff)
		if model.upper() != 'ARGON': intrinsic_dxdyz(directory, model, csize, frame, auv1, auv2, nmol, nsite, nm, nm, n0, phi, DIM, ow_coeff)

	return auv1, auv2, piv_n1, piv_n2


def build_surface(xmol, ymol, zmol, DIM, nmol, ncube, sigma, nm, n0, phi, vlim, zcom, tau, max_r):

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

	diag = np.diagflat([(ut.check_uv(int(j/(2*nm+1))-nm, int(j%(2*nm+1))-nm) * ((int(j/(2*nm+1))-nm)**2 * DIM[1] / DIM[0] + (int(j%(2*nm+1))-nm)**2 * DIM[0]/DIM[1])) for j in xrange((2*nm+1)**2)])

	diag *= 4 * np.pi**2 * phi

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
				zeta = xi(x, y, nm, nm, auv1, DIM)
				if len(piv_n1) < n0 and abs(zeta - z) <= tau:
					piv_n1.append(n)
					new_pivots1.append(n)
					mol_list.remove(n)
				elif abs(zeta - z) > 3.0 * sigma:
					mol_list.remove(n)					
			else:
				zeta = xi(x, y, nm, nm, auv2, DIM)
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

	if u >= 0: return np.cos(2 * np.pi * u * x / Lx)
	else: return np.sin(2 * np.pi * abs(u) * x / Lx)


def dfunction(x, u, Lx):

	if u >= 0: return - 2 * np.pi * u  / Lx * np.sin(2 * np.pi * u * x  / Lx)
	else: return 2 * np.pi * abs(u) / Lx * np.cos(2 * np.pi * abs(u) * x / Lx)


def ddfunction(x, u, Lx):

	return - 4 * np.pi**2 * u**2 / Lx**2 * function(x, u, Lx)


def xi(x, y, nm, qm, auv, DIM):

	zeta = 0
	for u in xrange(-qm, qm+1):
		for v in xrange(-qm, qm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)
			zeta += function(x, u, DIM[0]) * function(y, v, DIM[1]) * auv[j]
	return zeta


def dxyi(x, y, nm, qm, auv, DIM):

	dzx = 0
	dzy = 0
	for u in xrange(-qm, qm+1):
		for v in xrange(-qm, qm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)
			dzx += dfunction(x, u, DIM[0]) * function(y, v, DIM[1]) * auv[j]
			dzy += function(x, u, DIM[0]) * dfunction(y, v, DIM[1]) * auv[j]
	return dzx, dzy


def ddxyi(x, y, nm, qm, auv, DIM):


	ddzx = 0
	ddzy = 0
	for u in xrange(-qm, qm+1):
		for v in xrange(-qm, qm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)
			ddzx += ddfunction(x, u, DIM[0]) * function(y, v, DIM[1]) * auv[j]
			ddzy += function(x, u, DIM[0]) * ddfunction(y, v, DIM[1]) * auv[j]
	return ddzx, ddzy


def optimise_ns(directory, model, csize, nmol, nsite, nm, phi, vlim, ncube, DIM, COM, M, mol_sigma, start_ns, end_ns):

	if not os.path.exists('{}/DATA/ACOEFF'.format(directory)): os.mkdir('{}/DATA/ACOEFF'.format(directory))

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
			auv1, auv2, piv_n1, piv_n2 = intrinsic_surface(directory, model, csize, nsite, nmol, ncube, DIM, COM, nm, n0, phi, vlim, mol_sigma, M, frame, nframe, False)

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



def slice_area(auv1_2, auv2_2, nm, qm, DIM):
	"Obtain the intrinsic surface area"

        Axi1 = 0
	Axi2 = 0

	for u in xrange(-qm, qm+1):
                for v in xrange(-qm, qm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)
			dot_prod = np.pi**2  * (u**2/DIM[0]**2 + v**2/DIM[1]**2)

			f1_2 = ut.check_uv(u, v) * auv1_2[j]
			f2_2 = ut.check_uv(u, v) * auv2_2[j]

			Axi1 += f1_2 * dot_prod
			Axi2 += f2_2 * dot_prod

        return 1 + 0.5*Axi1, 1 + 0.5*Axi2


def gamma_q_auv(auv_2, nm, qm, DIM, T, q2_set):

	gamma_list = []
	gamma_hist = np.zeros(len(q2_set))
	gamma_count = np.zeros(len(q2_set))

	DIM = np.array(DIM) * 1E-10
	auv_2 *= 1E-20

	coeff = con.k * 1E3 * T

	for u in xrange(-qm, qm+1):
		for v in xrange(-qm, qm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)
			dot_prod = np.pi**2 * (u**2 * DIM[1] / DIM[0] + v**2 * DIM[0] / DIM[1])
			set_index = np.round(u**2*DIM[1]/DIM[0] + v**2*DIM[0]/DIM[1], 4)

			if set_index != 0:
				gamma = 1. / (ut.check_uv(u, v) * auv_2[j] * dot_prod)
				gamma_list.append(gamma)
				gamma_hist[q2_set == set_index] += gamma
				gamma_count[q2_set == set_index] += 1

	for i in xrange(len(q2_set)):
		if gamma_count[i] != 0: gamma_hist[i] *= 1. / gamma_count[i]

	return np.array(gamma_list) * coeff, gamma_hist * coeff


def gamma_q_f(f_2, nm, qm, DIM, T, q2_set):

	gamma_list = []
	gamma_hist = np.zeros(len(q2_set))
	gamma_count = np.zeros(len(q2_set))

	DIM = np.array(DIM) * 1E-10
	f_2 *= 1E-20

	coeff = con.k * 1E3 * T / (DIM[0] * DIM[1])

	for u in xrange(-qm, qm+1):
		for v in xrange(-qm, qm+1):
			j = (2 * nm + 1) * (u + nm) + (v + nm)
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


def intrinsic_positions(directory, model, csize, frame, auv1, auv2, natom, nmol, nsite, nm, QM, n0, phi, DIM, ow_pos):

	xat, yat, zat = ut.read_atom_positions(directory, model, csize, frame)
        xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)

	int_z_at_1 = np.zeros(natom)
	int_z_at_2 = np.zeros(natom)
	int_z_mol_1 = np.zeros(nmol)
	int_z_mol_2 = np.zeros(nmol)

	for qm in xrange(QM+1):
		sys.stdout.write("PROCESSING {} INTRINSIC POSITIONS {}: nm = {} qm = {}\r".format(directory, frame, nm, qm))
		sys.stdout.flush()

		if os.path.exists('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTZ_AT.txt'.format(directory, model.lower(), nm, qm, n0, int(1/phi + 0.5), frame)) and not ow_pos:
			with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTZ_AT.txt'.format(directory, model.lower(), nm, qm, n0, int(1/phi + 0.5), frame), 'r') as infile:
				int_z_at_1, int_z_at_2 = np.loadtxt(infile)
			with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTZ_MOL.txt'.format(directory, model.lower(), nm, qm, n0, int(1/phi + 0.5), frame), 'r') as infile:
				int_z_mol_1, int_z_mol_2 = np.loadtxt(infile)
		else:

			for n in xrange(natom):
				sys.stdout.write("PROCESSING {} INTRINSIC POSITIONS {}: nm = {} qm = {}  {} out of {}  atoms\r".format(directory, frame, nm, qm, n, natom))
		                sys.stdout.flush()

				m = n % nsite
				l = n / nsite

				if qm == 0:
					j = (2 * nm + 1) * nm + nm
					int_z_at_1[n] += function(xat[n], 0, DIM[0]) * function(yat[n], 0, DIM[1]) * auv1[j]
					int_z_at_2[n] += function(xat[n], 0, DIM[0]) * function(yat[n], 0, DIM[1]) * auv2[j]
					if m == 0:
						int_z_mol_1[l] += function(xmol[l], 0, DIM[0]) * function(ymol[l], 0, DIM[1]) * auv1[j]
						int_z_mol_2[l] += function(xmol[l], 0, DIM[0]) * function(ymol[l], 0, DIM[1]) * auv2[j]
				else:
					for u in [-qm, qm]:
						for v in xrange(-qm, qm+1):
							j = (2 * nm + 1) * (u + nm) + (v + nm)
							int_z_at_1[n] += function(xat[n], u, DIM[0]) * function(yat[n], v, DIM[1]) * auv1[j]
							int_z_at_2[n] += function(xat[n], u, DIM[0]) * function(yat[n], v, DIM[1]) * auv2[j]
							if m == 0:
								int_z_mol_1[l] += function(xmol[l], u, DIM[0]) * function(ymol[l], v, DIM[1]) * auv1[j]
								int_z_mol_2[l] += function(xmol[l], u, DIM[0]) * function(ymol[l], v, DIM[1]) * auv2[j]

					for u in xrange(-qm+1, qm):
						for v in [-qm, qm]:
							j = (2 * nm + 1) * (u + nm) + (v + nm)
							int_z_at_1[n] += function(xat[n], u, DIM[0]) * function(yat[n], v, DIM[1]) * auv1[j]
							int_z_at_2[n] += function(xat[n], u, DIM[0]) * function(yat[n], v, DIM[1]) * auv2[j]
							if m == 0:
								int_z_mol_1[l] += function(xmol[l], u, DIM[0]) * function(ymol[l], v, DIM[1]) * auv1[j]
								int_z_mol_2[l] += function(xmol[l], u, DIM[0]) * function(ymol[l], v, DIM[1]) * auv2[j]

			with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTZ_AT.txt'.format(directory, model.lower(), nm, qm, n0, int(1/phi + 0.5), frame), 'w') as outfile:
				np.savetxt(outfile, (int_z_at_1, int_z_at_2), fmt='%-12.6f')
			with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTZ_MOL.txt'.format(directory, model.lower(), nm, qm, n0, int(1/phi + 0.5), frame), 'w') as outfile:
				np.savetxt(outfile, (int_z_mol_1, int_z_mol_2), fmt='%-12.6f')

	return int_z_at_1, int_z_at_2, int_z_mol_1, int_z_mol_2

def intrinsic_dxdyz(directory, model, csize, frame, auv1, auv2, nmol, nsite, nm, QM, n0, phi, DIM, ow_dxdyz):

        xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)
        dxdyz_mol = [np.zeros(nmol) for n in range(4)]

        for qm in xrange(QM+1):
		sys.stdout.write("PROCESSING {} INTRINSIC DERIVATIVES {}: nm = {} qm = {}\r".format(directory, frame, nm, qm))
		sys.stdout.flush()

		if os.path.exists('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTDXDY_MOL.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame)) and not ow_dxdyz:
			with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTDXDY_MOL.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame), 'r') as infile:
				dxdyz_mol = np.loadtxt(infile)
		else:
		        for n in xrange(nmol):
		                sys.stdout.write("PROCESSING {} INTRINSIC DERIVATIVES {}: nm = {} qm = {}  {} out of {} molecules\r".format(directory, frame, nm, qm, n, nmol))
		                sys.stdout.flush()

		                if qm == 0:
					j = (2 * nm + 1) * nm + nm
					dxdyz_mol[0][n] += dfunction(xmol[n], 0, DIM[0]) * function(ymol[n], 0, DIM[1]) * auv1[j]
					dxdyz_mol[1][n] += function(xmol[n], 0, DIM[0]) * dfunction(ymol[n], 0, DIM[1]) * auv1[j]
					dxdyz_mol[2][n] += dfunction(xmol[n], 0, DIM[0]) * function(ymol[n], 0, DIM[1]) * auv2[j]
					dxdyz_mol[3][n] += function(xmol[n], 0, DIM[0]) * dfunction(ymol[n], 0, DIM[1]) * auv2[j]

		                else:
					for u in [-qm, qm]:
						for v in xrange(-qm, qm+1):
							j = (2 * nm + 1) * (u + nm) + (v + nm)
							dxdyz_mol[0][n] += dfunction(xmol[n], u, DIM[0]) * function(ymol[n], v, DIM[1]) * auv1[j]
							dxdyz_mol[1][n] += function(xmol[n], u, DIM[0]) * dfunction(ymol[n], v, DIM[1]) * auv1[j]
							dxdyz_mol[2][n] += dfunction(xmol[n], u, DIM[0]) * function(ymol[n], v, DIM[1]) * auv2[j]
							dxdyz_mol[3][n] += function(xmol[n], u, DIM[0]) * dfunction(ymol[n], v, DIM[1]) * auv2[j]

					for u in xrange(-qm+1, qm):
						for v in [-qm, qm]:
							j = (2 * nm + 1) * (u + nm) + (v + nm)
							dxdyz_mol[0][n] += dfunction(xmol[n], u, DIM[0]) * function(ymol[n], v, DIM[1]) * auv1[j]
							dxdyz_mol[1][n] += function(xmol[n], u, DIM[0]) * dfunction(ymol[n], v, DIM[1]) * auv1[j]
							dxdyz_mol[2][n] += dfunction(xmol[n], u, DIM[0]) * function(ymol[n], v, DIM[1]) * auv2[j]
							dxdyz_mol[3][n] += function(xmol[n], u, DIM[0]) * dfunction(ymol[n], v, DIM[1]) * auv2[j]

			with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTDXDY_MOL.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame), 'w') as outfile:
		       		np.savetxt(outfile, (dxdyz_mol), fmt='%-12.10f')

	return dxdyz_mol

def intrinsic_density(directory, COM, model, csize, nm, qm, n0, phi, frame, nslice, nsite, AT, DIM, M, ow_count):
	"Saves atom, mol and mass intrinsic profiles of trajectory frame" 
	
	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	if os.path.exists('{}/DATA/INTDEN/{}_{}_{}_{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), frame)) and not ow_count:
		try:
			with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), frame)) as infile:
				count_array = np.loadtxt(infile)
			for i in xrange(3 + n_atom_types): count_array[i]
		except IndexError: ow_count = True
	else: ow_count = True

	if ow_count:

		xat, yat, zat = ut.read_atom_positions(directory, model, csize, frame)
		xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)
		xR, yR, zR = COM[frame]

		count_array = [np.zeros(nslice) for n in range(3 + n_atom_types)]

		Aslice = DIM[0] * DIM[1]
		natom = len(xat)
		nmol = len(xmol)

		try:
			with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTZ_AT.txt'.format(directory, model.lower(), nm, qm, n0, int(1/phi + 0.5), frame), 'r') as infile:
				int_z_at_1, int_z_at_2 = np.loadtxt(infile)
			with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTZ_MOL.txt'.format(directory, model.lower(), nm, qm, n0, int(1/phi + 0.5), frame), 'r') as infile:
				int_z_mol_1, int_z_mol_2 = np.loadtxt(infile)
		except:
			with file('{}/DATA/ACOEFF/{}_{}_{}_{}_{}_INTCOEFF.txt'.format(directory, model.lower(), nm, n0, int(1./phi + 0.5), frame), 'r') as infile: 
				auv1, auv2 = np.loadtxt(infile)
			int_z_at_1, int_z_at_2, int_z_mol_1, int_z_mol_2 = intrinsic_positions(directory, model, csize, frame, auv1, auv2, natom, nmol, nsite, nm, qm, n0, phi, DIM, False)

	       	for n in xrange(natom):
			sys.stdout.write("PROCESSING {} INTRINSIC DENSITY {}: nm = {} qm = {}  {} out of {}  atoms\r".format(directory, frame, nm, qm, n, natom) )
			sys.stdout.flush()

			m = n % nsite
			at_type = AT[m]
	
			x = xat[n]
			y = yat[n]
			z = zat[n] - zR

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

				int_z1 = int_z_mol_1[n/nsite]
				int_z2 = int_z_mol_2[n/nsite]

				z1 = z - int_z1
				z2 = -z + int_z2

				index1_mol = int((z1 + DIM[2]/2.) * nslice / (DIM[2])) % nslice
				index2_mol = int((z2 + DIM[2]/2.) * nslice / (DIM[2])) % nslice

				count_array[-2][index1_mol] += 1
				count_array[-1][index2_mol] += 1

		count_array[-1] = count_array[-1][::-1]

		with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), frame), 'w') as outfile:
			np.savetxt(outfile, (count_array), fmt='%-12.6f')

	return count_array


def effective_density(directory, model, nslice, nframe, DIM, nm, qm, n0, phi, av_density_array):

	print "\nBUILDING SLAB DENSITY PLOT {}/DATA/INTDEN/ CWDEN.txt".format(directory)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice)
	lslice = DIM[2] / nslice

	av_cw_den_1 = np.zeros(nslice)
        av_cw_den_2 = np.zeros(nslice)

	mean_auv1 = np.zeros(nframe)
	mean_auv2 = np.zeros(nframe)

	av_auv1_2 = np.zeros((2*nm+1)**2)
	av_auv2_2 = np.zeros((2*nm+1)**2)

	for frame in xrange(nframe):
		sys.stdout.write("PROCESSING {} out of {} IMAGES\r".format(frame, nframe) )
		sys.stdout.flush()

		with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_{}_{}_COUNT.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), frame)) as infile:
			count_array = np.loadtxt(infile)
		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_{}_INTCOEFF.txt'.format(directory, model.lower(), nm, n0, int(1/phi + 0.5), frame), 'r') as infile:
			auv1, auv2 = np.loadtxt(infile)		

		mean_auv1[frame] += auv1[len(auv1)/2]
                mean_auv2[frame] += auv2[len(auv2)/2]

		auv1_2 = auv1**2
		auv2_2 = auv2**2

                av_auv1_2 += auv1_2 / nframe
                av_auv2_2 += auv2_2 / nframe
		
	Delta1 = (ut.sum_auv_2(av_auv1_2, nm, qm) - np.mean(mean_auv1)**2)
	Delta2 = (ut.sum_auv_2(av_auv2_2, nm, qm) - np.mean(mean_auv2)**2)

	deltas = [Delta1, Delta2, Delta1 + np.mean(mean_auv1)**2 - av_auv1_2[len(auv1)/2], Delta2 + np.mean(mean_auv2)**2 - av_auv2_2[len(auv1)/2]]
	centres = [np.mean(mean_auv1), np.mean(mean_auv2), np.mean(mean_auv1), np.mean(mean_auv2)]
	arrays = [av_density_array[-4], av_density_array[-3], av_density_array[-4], av_density_array[-3]]

	cw_arrays = ut.gaussian_smoothing(arrays, centres, deltas, DIM, nslice)
	
	return cw_arrays[0], cw_arrays[1], Delta1, Delta2


def intrinsic_R_tensors(directory, model, csize, frame, nslice, COM, DIM, nsite, nm, qm, n0, phi, ow_R):

	if os.path.exists('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_{}_ODIST1.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1./phi + 0.5), frame)) and not ow_R: 
		try:
			with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_{}_ODIST1.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1./phi+0.5), frame), 'r') as infile:
				temp_int_O1 = np.loadtxt(infile)
			with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_{}_ODIST2.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1./phi+0.5), frame), 'r') as infile:
				temp_int_O2 = np.loadtxt(infile)

		except Exception: ow_R = True
	else: ow_R = True

	if ow_R:

		xat, yat, zat = ut.read_atom_positions(directory, model, csize, frame)
		xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)
		xR, yR, zR = COM[frame]

		nmol = len(xmol)

		temp_int_O1 = np.zeros((nslice, 9))
		temp_int_O2 = np.zeros((nslice, 9))

		with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTZ_MOL.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame), 'r') as infile:
			int_z_mol_1, int_z_mol_2 = np.loadtxt(infile)
		try:	
			with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTDXDY_MOL.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame), 'r') as infile:
				dxdyz_mol = np.loadtxt(infile)
		except:
			with file('{}/DATA/ACOEFF/{}_{}_{}_{}_{}_INTCOEFF.txt'.format(directory, model.lower(), nm, n0, int(1./phi + 0.5), frame), 'r') as infile: 
				auv1, auv2 = np.loadtxt(infile)
			dxdyz_mol = intrinsic_dxdyz(directory, model, csize, frame, auv1, auv2, nmol, nsite, nm, nm, n0, phi, DIM, False)

		for j in xrange(nmol):
			sys.stdout.write("PROCESSING {} ODIST {}: {} out of {}  molecules\r".format(directory, frame, j, nmol))
			sys.stdout.flush()

			molecule = np.zeros((nsite, 3))

			dzx1 = dxdyz_mol[0][j]
			dzy1 = dxdyz_mol[1][j]
			dzx2 = dxdyz_mol[2][j]
			dzy2 = dxdyz_mol[3][j]

			for l in xrange(nsite):
				molecule[l][0] = xat[j*nsite+l]
				molecule[l][1] = yat[j*nsite+l]
				molecule[l][2] = zat[j*nsite+l]

			zeta1 = zmol[j] - zR - int_z_mol_1[j] 
			zeta2 = - zmol[j] + zR + int_z_mol_2[j]

			"""NORMAL Z AXIS"""

			O = ut.local_frame_molecule(molecule, model)
			if O[2][2] < -1: O[2][2] = -1.0
			elif O[2][2] > 1: O[2][2] = 1.0

			""" INTRINSIC SURFACE DERIVATIVE """
			#"""
			T = ut.local_frame_surface(dzx1, dzy1, -1)
			R1 = np.dot(O, np.linalg.inv(T))
			if R1[2][2] < -1: R1[2][2] = -1.0
			elif R1[2][2] > 1: R1[2][2] = 1.0

			T = ut.local_frame_surface(dzx2, dzy2, 1)
			R2 = np.dot(O, np.linalg.inv(T))
			if R2[2][2] < -1: R2[2][2] = -1.0
			elif R2[2][2] > 1: R2[2][2] = 1.0
			#"""

			int_index1 = int((zeta1 + DIM[2]/2) * nslice / DIM[2]) % nslice
			int_index2 = int((zeta2 + DIM[2]/2) * nslice / DIM[2]) % nslice

			for k in xrange(3):
				for l in xrange(3):
					index2 = k * 3 + l 
					temp_int_O1[int_index1][index2] += R1[k][l]**2
					temp_int_O2[int_index2][index2] += R2[k][l]**2
		
		with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_{}_ODIST1.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1./phi + 0.5), frame), 'w') as outfile:
			np.savetxt(outfile, temp_int_O1, fmt='%-12.10f')
		with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_{}_ODIST2.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1./phi + 0.5), frame), 'w') as outfile:
			np.savetxt(outfile, temp_int_O2, fmt='%-12.10f')

	return temp_int_O1, temp_int_O2


def intrinsic_mol_angles(directory, model, csize, frame, nslice, npi, nmol, COM, DIM, nsite, nm, qm, n0, phi, ow_angle):

	dpi = np.pi / npi

	temp_int_P_z_theta_phi_1 = np.zeros((nslice, npi, npi*2))
	temp_int_P_z_theta_phi_2 = np.zeros((nslice, npi, npi*2))

	if os.path.exists('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_ANGLE1.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame)) and not ow_angle:	
		try:
			with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_ANGLE1.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame), 'r') as infile:
				zeta_array1, int_theta1, int_phi1, int_varphi1 = np.loadtxt(infile)
			with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_ANGLE2.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame), 'r') as infile:
				zeta_array2, int_theta2, int_phi2, int_varphi2 = np.loadtxt(infile)

			for j in xrange(nmol):
				z = zeta_array1[j] + DIM[2]/2.
				index1 = int(z * nslice / DIM[2]) % nslice
				index2 = int(int_theta1[j] /dpi)
				index3 = int((int_phi1[j] + np.pi / 2.) / dpi) 

				try: temp_int_P_z_theta_phi_1[index1][index2][index3] += 1
				except IndexError: pass

				z = zeta_array2[j] + DIM[2]/2.
				index1 = int(z * nslice / DIM[2]) % nslice
				index2 = int(int_theta2[j] / dpi)
				index3 = int((int_phi2[j] + np.pi / 2.) / dpi)  

				try: temp_int_P_z_theta_phi_2[index1][index2][index3] += 1
				except IndexError: pass

		except Exception: ow_angle = True

	else: ow_angle = True

	if ow_angle:

		zeta_array1 = np.zeros(nmol)
		int_theta1 = np.zeros(nmol)
		int_phi1 = np.zeros(nmol)
		int_varphi1 = np.zeros(nmol)

		zeta_array2 = np.zeros(nmol)
		int_theta2 = np.zeros(nmol)
		int_phi2 = np.zeros(nmol)
		int_varphi2 =np.zeros(nmol)

		with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTZ_MOL.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame), 'r') as infile:
			int_z_mol_1, int_z_mol_2 = np.loadtxt(infile)
		try:	
			with file('{}/DATA/INTPOS/{}_{}_{}_{}_{}_{}_INTDXDY_MOL.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame), 'r') as infile:
				dxdyz_mol = np.loadtxt(infile)
		except:
			with file('{}/DATA/ACOEFF/{}_{}_{}_{}_{}_INTCOEFF.txt'.format(directory, model.lower(), nm, n0, int(1./phi + 0.5), frame), 'r') as infile: 
				auv1, auv2 = np.loadtxt(infile)
			dxdyz_mol = intrinsic_dxdyz(directory, model, csize, frame, auv1, auv2, nmol, nsite, nm, nm, n0, phi, DIM, False)

		xat, yat, zat = ut.read_atom_positions(directory, model, csize, frame)
		xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)
		xR, yR, zR = COM[frame]

		for j in xrange(nmol):
			sys.stdout.write("PROCESSING {} INTRINSIC ANGLES {}: nm = {} qm = {}  {} out of {}  molecules\r".format(directory, frame, nm, qm, j, nmol) )
			sys.stdout.flush()

			molecule = np.zeros((nsite, 3))

			for l in xrange(nsite):
				molecule[l][0] = xat[j*nsite+l]
				molecule[l][1] = yat[j*nsite+l]
				molecule[l][2] = zat[j*nsite+l]

			zeta1 = zmol[j] - zR - int_z_mol_1[j] 
			zeta2 = - zmol[j] + zR + int_z_mol_2[j]

			dzx1 = dxdyz_mol[0][j]
			dzy1 = dxdyz_mol[1][j]
			dzx2 = dxdyz_mol[2][j]
			dzy2 = dxdyz_mol[3][j]

			"""NORMAL Z AXIS"""

			O = ut.local_frame_molecule(molecule, model)
			if O[2][2] < -1: O[2][2] = -1.0
		        elif O[2][2] > 1: O[2][2] = 1.0

			#print "\n{} {} {}".format(np.arccos(O[2][2]), np.arctan(-O[2][0]/O[2][1]), np.arctan(O[0][2]/O[1][2]))
		
			""" INTRINSIC SURFACE """
			"""
			zeta_array1[j] = zeta1
			int_theta1[j] = np.arccos(O[2][2])
			int_phi1][j] = np.arctan2(-O[2][0],O[2][1])
			int_varphi1[j] = np.arctan2(O[0][2],O[1][2])

			zeta_array2[j] = zeta2
			int_theta2[j] = np.arccos(O[2][2])
			int_phi2[j] = np.arctan2(-O[2][0],O[2][1])
			int_varphi2[j] = np.arctan2(O[0][2],O[1][2])
			"""
			""" INTRINSIC SURFACE DERIVATIVE """

			T = ut.local_frame_surface(dzx1, dzy1, -1)
			R1 = np.dot(O, np.linalg.inv(T))
			if R1[2][2] < -1: R1[2][2] = -1.0
			elif R1[2][2] > 1: R1[2][2] = 1.0
			zeta_array1[j] = zeta1
			int_theta1[j] = np.arccos(R1[2][2])
			int_phi1[j] = (np.arctan(-R1[2][0] / R1[2][1]))
			int_varphi1[j] = (np.arctan(R1[0][2] / R1[1][2]))

			T = ut.local_frame_surface(dzx2, dzy2, 1)
			R2 = np.dot(O, np.linalg.inv(T))
			if R2[2][2] < -1: R2[2][2] = -1.0
			elif R2[2][2] > 1: R2[2][2] = 1.0
			zeta_array2[j] = zeta2
			int_theta2[j] = np.arccos(R2[2][2])
			int_phi2[j] = (np.arctan(-R2[2][0] / R2[2][1]))
			int_varphi2[j] = (np.arctan(R2[0][2] / R2[1][2]))

			z = zeta_array1[j] + DIM[2]/2.
			index1 = int(z * nslice / DIM[2]) % nslice
			index2 = int(int_theta1[j] /dpi)
			index3 = int((int_phi1[j] + np.pi / 2.) / dpi) 

			try: temp_int_P_z_theta_phi_1[index1][index2][index3] += 1
			except IndexError: pass

			z = zeta_array2[j] + DIM[2]/2.
			index1 = int(z * nslice / DIM[2]) % nslice
			index2 = int(int_theta2[j] / dpi)
			index3 = int((int_phi2[j] + np.pi / 2.) / dpi)  

			try: temp_int_P_z_theta_phi_2[index1][index2][index3] += 1
			except IndexError: pass


		with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_ANGLE1.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame), 'w') as outfile:
			np.savetxt(outfile, (zeta_array1, int_theta1, int_phi1, int_varphi1), fmt='%-12.6f')
		with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_ANGLE2.txt'.format(directory, model.lower(), nm, qm, n0, int(1./phi + 0.5), frame), 'w') as outfile:
			np.savetxt(outfile, (zeta_array2, int_theta2, int_phi2, int_varphi2), fmt='%-12.6f')


	return temp_int_P_z_theta_phi_1, temp_int_P_z_theta_phi_2


def intrinsic_angle_dist(nslice, npi, int_P_z_theta_phi_1, int_P_z_theta_phi_2):

	print "BUILDING ANGLE DISTRIBUTIONS"

	dpi = np.pi / npi

	print ""
	print "NORMALISING GRID"
	for index1 in xrange(nslice): 
		if np.sum(int_P_z_theta_phi_1[index1]) != 0:
			int_P_z_theta_phi_1[index1] = int_P_z_theta_phi_1[index1] / np.sum(int_P_z_theta_phi_1[index1])
		if np.sum(int_P_z_theta_phi_2[index1]) != 0:
			int_P_z_theta_phi_2[index1] = int_P_z_theta_phi_2[index1] / np.sum(int_P_z_theta_phi_2[index1])

	int_P_z_phi_theta_1 = np.rollaxis(np.rollaxis(int_P_z_theta_phi_1, 2), 1)
	int_P_z_phi_theta_2 = np.rollaxis(np.rollaxis(int_P_z_theta_phi_2, 2), 1)
	
	X_theta = np.arange(0, np.pi, dpi)
	X_phi = np.arange(-np.pi / 2, np.pi / 2, dpi)

	int_av_theta1 = np.zeros(nslice)
        int_av_phi1 = np.zeros(nslice)
	int_P11 = np.zeros(nslice)
	int_P21 = np.zeros(nslice)

	int_av_theta2 = np.zeros(nslice)
	int_av_phi2 = np.zeros(nslice)
	int_P12 = np.zeros(nslice)
	int_P22 = np.zeros(nslice)
	
	print "BUILDING AVERAGE ANGLE PROFILES"

	for index1 in xrange(nslice):
		sys.stdout.write("PROCESSING AVERAGE ANGLE PROFILES {} out of {} slices\r".format(index1, nslice) )
		sys.stdout.flush() 

		for index2 in xrange(npi):

			int_av_theta1[index1] += np.sum(int_P_z_theta_phi_1[index1][index2]) * X_theta[index2] 
			int_P11[index1] += np.sum(int_P_z_theta_phi_1[index1][index2]) * np.cos(X_theta[index2])
			int_P21[index1] += np.sum(int_P_z_theta_phi_1[index1][index2]) * 0.5 * (3 * np.cos(X_theta[index2])**2 - 1)

			int_av_theta2[index1] += np.sum(int_P_z_theta_phi_2[index1][index2]) * X_theta[index2] 
			int_P12[index1] += np.sum(int_P_z_theta_phi_2[index1][index2]) * np.cos(X_theta[index2])
			int_P22[index1] += np.sum(int_P_z_theta_phi_2[index1][index2]) * 0.5 * (3 * np.cos(X_theta[index2])**2 - 1)

			int_av_phi1[index1] += np.sum(int_P_z_phi_theta_1[index1][index2]) * (X_phi[index2]) 
			int_av_phi2[index1] += np.sum(int_P_z_phi_theta_2[index1][index2]) * (X_phi[index2])

		if int_av_theta1[index1] == 0: 
			int_av_theta1[index1] += np.pi / 2.
			int_av_phi1[index1] += np.pi / 4.
		if int_av_theta2[index1] == 0: 
			int_av_theta2[index1] += np.pi / 2.
			int_av_phi2[index1] += np.pi / 4.

	a_dist = (int_av_theta1, int_av_phi1, int_P11, int_P21, int_av_theta2, int_av_phi2, int_P12, int_P22)
	
	return a_dist


def intrinsic_polarisability(nslice, a, count_int_O1, count_int_O2, av_int_O1, av_int_O2):

	int_axx1 = np.zeros(nslice)
	int_azz1 = np.zeros(nslice)

	int_axx2 = np.zeros(nslice)
	int_azz2 = np.zeros(nslice)

	for n in xrange(nslice):
		if count_int_O1[n] != 0:
			av_int_O1[n] *= 1./ count_int_O1[n]
			for j in xrange(3):
				int_axx1[n] += a[j] * 0.5 * (av_int_O1[n][j] + av_int_O1[n][j+3]) 
				int_azz1[n] += a[j] * av_int_O1[n][j+6] 

		else: 					
			int_axx1[n] = np.mean(a)					
			int_azz1[n] = np.mean(a)

		if count_int_O2[n] != 0:
			av_int_O2[n] *= 1./ count_int_O2[n]
			for j in xrange(3):
				int_axx2[n] += a[j] * 0.5 * (av_int_O2[n][j] + av_int_O2[n][j+3]) 
				int_azz2[n] += a[j] * av_int_O2[n][j+6] 
		else: 
			int_axx2[n] = np.mean(a)					
			int_azz2[n] = np.mean(a)

	polar = (int_axx1, int_azz1, int_axx2, int_azz2)

	return polar


def intrinsic_profile(directory, model, csize, nframe, natom, nmol, nsite, AT, M, a_type, mol_sigma, COM, DIM, nslice, ncube, nm, QM, n0, phi, npi, vlim, ow_profile, ow_coeff, ow_pos, ow_dxdyz, ow_count, ow_angle, ow_polar):

	lslice = DIM[2] / nslice
	Aslice = DIM[0]*DIM[1]
	Vslice = DIM[0]*DIM[1]*lslice
	Acm = 1E-8
	ur = 1
	Z2 = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	a = ut.get_polar_constants(model, a_type)

	mean_auv1 = np.zeros(nframe)
	mean_auv2 = np.zeros(nframe)

	av_auv1_2 = np.zeros((2*nm+1)**2)
	av_auv2_2 = np.zeros((2*nm+1)**2)

	for frame in xrange(nframe):

		auv1, auv2, piv_n1, piv_n2 = intrinsic_surface(directory, model, csize, nsite, nmol, ncube, DIM, COM, nm, n0, phi, vlim, mol_sigma, M, frame, nframe, ow_coeff)	

		if ow_pos: intrinsic_positions(directory, model, csize, frame, auv1, auv2, natom, nmol, nsite, nm, nm, n0, phi, DIM, ow_pos)
		if model.upper() != 'ARGON': 
			if ow_dxdyz: intrinsic_dxdyz(directory, model, csize, frame, auv1, auv2, nmol, nsite, nm, nm, n0, phi, DIM, ow_coeff)		

		mean_auv1[frame] += auv1[len(auv1)/2]
                mean_auv2[frame] += auv2[len(auv2)/2]

		auv1_2 = auv1**2
		auv2_2 = auv2**2

                av_auv1_2 += auv1_2 / nframe
                av_auv2_2 += auv2_2 / nframe

	print ""

	for qm in QM:
		if not os.path.exists('{}/DATA/INTDEN/{}_{}_{}_{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), nframe)) or ow_profile:

			count_int_O1 = np.zeros(nslice)
			count_int_O2 = np.zeros(nslice)

			av_int_O1 = np.zeros((nslice, 9))
			av_int_O2 = np.zeros((nslice, 9))

			int_P_z_theta_phi_1 = np.zeros((nslice, npi, npi*2))
			int_P_z_theta_phi_2 = np.zeros((nslice, npi, npi*2))

			int_av_varphi1 = np.zeros(nslice)
			int_av_varphi2 = np.zeros(nslice)

			av_density_array = [np.zeros(nslice) for n in range(5 + n_atom_types)]

			for frame in xrange(nframe):
				sys.stdout.write("PROCESSING FRAME {}\r".format(frame))
				sys.stdout.flush()

				int_count_array = intrinsic_density(directory, COM, model, csize,  nm, qm, n0, phi, frame, nslice, nsite, AT, DIM, M, ow_count)

				for i in xrange(3 + n_atom_types): av_density_array[i] += int_count_array[i] / (nframe * Vslice)

				count_int_O1 += int_count_array[-2]
				count_int_O2 += int_count_array[-1][::-1]

				if model.upper() != 'ARGON':

					temp_int_P_z_theta_phi_1, temp_int_P_z_theta_phi_2 = intrinsic_mol_angles(directory, model, csize, frame, nslice, npi, nmol, COM, DIM, nsite, nm, qm, n0, phi, ow_angle)

					int_P_z_theta_phi_1 += temp_int_P_z_theta_phi_1
					int_P_z_theta_phi_2 += temp_int_P_z_theta_phi_2

					temp_int_O1, temp_int_O2 = intrinsic_R_tensors(directory, model, csize, frame, nslice, COM, DIM, nsite, nm, qm, n0, phi, ow_polar)

					av_int_O1 += temp_int_O1
					av_int_O2 += temp_int_O2

			with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), nframe), 'w') as outfile:
				np.savetxt(outfile, (av_density_array), fmt='%-12.6f')

			Delta1 = (ut.sum_auv_2(av_auv1_2, nm, qm) - np.mean(mean_auv1)**2)
			Delta2 = (ut.sum_auv_2(av_auv2_2, nm, qm) - np.mean(mean_auv2)**2)

			av_density_array[0] = av_density_array[0] / (con.N_A * Acm**3)

			#cw_den_1, cw_den_2, Delta1, Delta2 = effective_density(directory, model, nslice, nframe, DIM, nm, qm, n0, phi, av_density_array, ow_wden)

			if model.upper() == 'ARGON':

				int_axx = np.ones(nslice) * a
				int_azz = np.ones(nslice) * a
				int_axx1 = np.ones(nslice) * a 
				int_azz1 = np.ones(nslice) * a
				int_axx2 = np.ones(nslice) * a 
				int_azz2 = np.ones(nslice) * a
			else:
		
				int_axx1, int_azz1, int_axx2, int_azz2 = intrinsic_polarisability(nslice, a, count_int_O1, count_int_O2, av_int_O1, av_int_O2)
				int_av_theta1, int_av_phi1, int_P11, int_P21, int_av_theta2, int_av_phi2, int_P12, int_P22 = intrinsic_angle_dist(nslice, npi, int_P_z_theta_phi_1, int_P_z_theta_phi_2)

				with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_{}_{}_EUL1.txt'.format(directory, model.lower(), a_type, nslice, nm, qm, n0, int(1/phi + 0.5), nframe), 'w') as outfile:
					np.savetxt(outfile, (int_axx1, int_azz1, int_av_theta1, int_av_phi1,int_av_varphi1,int_P11, int_P21), fmt='%-12.6f')
				with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_{}_{}_EUL2.txt'.format(directory, model.lower(), a_type, nslice, nm, qm, n0, int(1/phi + 0.5), nframe), 'w') as outfile:
					np.savetxt(outfile, (int_axx2[::-1], int_azz2[::-1], int_av_theta2, int_av_phi2, int_av_varphi2, int_P12, int_P22), fmt='%-12.6f')

			mol_int_den = 0.5 * (av_density_array[-4] + av_density_array[-3][::-1])

			int_axx = 0.5 * (int_axx1 + int_axx2[::-1])
			int_azz = 0.5 * (int_azz1 + int_azz2[::-1])

			rho_axx =  np.array([mol_int_den[n] * int_axx[n] for n in range(nslice)])
			rho_azz =  np.array([mol_int_den[n] * int_azz[n] for n in range(nslice)])

			int_exx = np.array([(1 + 8 * np.pi / 3. * rho_axx[n]) / (1 - 4 * np.pi / 3. * rho_axx[n]) for n in range(nslice)])
			int_ezz = np.array([(1 + 8 * np.pi / 3. * rho_azz[n]) / (1 - 4 * np.pi / 3. * rho_azz[n]) for n in range(nslice)])

			int_no = np.sqrt(ur * int_exx)
			int_ni = np.sqrt(ur * int_ezz)

			centres = [np.mean(mean_auv1), np.mean(mean_auv2)] + list(np.ones(9) * np.mean(mean_auv1))
			deltas = [Delta1, Delta2] + list(np.ones(9) * 0.5 * (Delta1 + Delta2))

			arrays = [av_density_array[-4], av_density_array[-3], mol_int_den, int_axx, int_azz, rho_axx, rho_azz, int_exx, int_ezz, int_no, int_ni]

			cw_arrays = ut.gaussian_smoothing(arrays, centres, deltas, DIM, nslice)

			av_density_array[-2] += cw_arrays[0]
			av_density_array[-1] += cw_arrays[1]

			cw_exx1 = np.array([(1 + 8 * np.pi / 3. * cw_arrays[2][n] * cw_arrays[3][n]) / (1 - 4 * np.pi / 3. * cw_arrays[2][n] * cw_arrays[3][n]) for n in range(nslice)])
			cw_ezz1 = np.array([(1 + 8 * np.pi / 3. * cw_arrays[2][n] * cw_arrays[4][n]) / (1 - 4 * np.pi / 3. * cw_arrays[2][n] * cw_arrays[4][n]) for n in range(nslice)])

			cw_exx2 = np.array([(1 + 8 * np.pi / 3. * cw_arrays[5][n]) / (1 - 4 * np.pi / 3. * cw_arrays[5][n]) for n in range(nslice)])
			cw_ezz2 = np.array([(1 + 8 * np.pi / 3. * cw_arrays[6][n]) / (1 - 4 * np.pi / 3. * cw_arrays[6][n]) for n in range(nslice)])

			"""
			cw_int_no1 = np.sqrt(ur * cw_int_exx1)
			anis = np.array([1 - (cw_int_ezz1[n] - cw_int_exx1[n]) * np.sin(angle)**2 / cw_int_ezz1[n] for n in range(nslice)])
			cw_int_ne1 = np.array([np.sqrt(ur * cw_int_exx1[n] / anis[n]) for n in range(nslice)])
			cw_int_ni1 = np.sqrt(ur * cw_int_ezz1)

			cw_int_no2 = np.sqrt(ur * cw_int_exx2)
			anis = np.array([1 - (cw_int_ezz2[n] - cw_int_exx2[n]) * np.sin(angle)**2 / cw_int_ezz2[n] for n in range(nslice)])
			cw_int_ne2 = np.array([np.sqrt(ur * cw_int_exx2[n] / anis[n]) for n in range(nslice)])
			cw_int_ni2 = np.sqrt(ur * cw_int_ezz2)
			"""

			"""
			plt.plot(no_sm)
			plt.plot(np.sqrt(cw_exx1))
			plt.plot(np.sqrt(cw_exx2))
			plt.plot(np.sqrt(cw_arrays[5]))
			plt.plot(cw_arrays[7])
			plt.show()
			"""

			print '\n'
			print "WRITING TO FILE... nm = {}  qm = {}  var1 = {}  var2 = {}".format(nm, qm, Delta1, Delta2)

			with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nm, qm, n0, int(1/phi + 0.5), nframe), 'w') as outfile:
				np.savetxt(outfile, (av_density_array), fmt='%-12.6f')
			with file('{}/DATA/INTDIELEC/{}_{}_{}_{}_{}_{}_{}_{}_DIE.txt'.format(directory, model.lower(), a_type, nslice, nm, qm, n0, int(1./phi + 0.5), nframe), 'w') as outfile:
				np.savetxt(outfile, (int_exx, int_ezz), fmt='%-12.6f')
			with file('{}/DATA/INTDIELEC/{}_{}_{}_{}_{}_{}_{}_{}_CWDIE.txt'.format(directory, model.lower(), a_type, nslice, nm, qm, n0, int(1./phi + 0.5), nframe), 'w') as outfile:
				np.savetxt(outfile, (cw_exx1, cw_ezz1, cw_exx2, cw_ezz2, cw_arrays[7], cw_arrays[8], cw_arrays[9]**2, cw_arrays[10]**2), fmt='%-12.6f')
			with file('{}/DATA/ELLIP/{}_{}_{}_{}_{}_{}_{}_{}_ELLIP_NO.txt'.format(directory, model.lower(), a_type, nslice, nm, qm, n0, int(1./phi + 0.5), nframe), 'w') as outfile:
				np.savetxt(outfile, (np.sqrt(cw_exx1), np.sqrt(cw_exx2), np.sqrt(cw_arrays[7]), cw_arrays[9]), fmt='%-12.6f')


		print "INTRINSIC SAMPLING METHOD {} {} {} {} {} {} COMPLETE\n".format(directory, model.upper(), nm, qm, n0, phi)


