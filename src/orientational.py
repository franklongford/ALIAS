"""
*************** ORIENTATIONAL ANALYSIS MODULE *******************

Calculates Euler angle profile and polarisability of system

***************************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 24/11/2016 by Frank Longford
"""

import numpy as np
import scipy as sp
import subprocess, time, sys, os, math, copy
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
import intrinsic_surface as IS


def euler_profile(traj, root, nimage, nslice, nmol, model, csize, suffix, DIM, nsite, a_type, nm, nxy, ow_P):

	print ""
	print "CALCULATING ORIENTATIONS OF {} {} SIZE {}\n".format(model.upper(), suffix.upper(), csize)	

	MOLECULES = np.zeros((nimage, nmol, nsite, 3))
	nsite, AT, Q, M, LJ = ut.get_param(model)
	npi = 50
	if model.upper() == 'METHANOL': com = 'COM'
	else: com = '0'

	X = np.linspace(0, DIM[0], nxy)
	Y = np.linspace(0, DIM[1], nxy)
	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice)

	with file('{}/DATA/DEN/{}_{}_{}_COM.txt'.format(root, model.lower(), csize, nimage), 'r') as infile:
		xR, yR, zR = np.loadtxt(infile)
		if nimage == 1: zR = [zR]

	with file('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, nimage), 'r') as infile:
		av_mass_den, av_atom_den, av_mol_den, av_H_den = np.loadtxt(infile)

	with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, nm, nimage), 'r') as infile:
		int_av_mass_den, int_av_atom_den, int_av_mol_den, int_av_H_den, w_den_1, w_den_2 = np.loadtxt(infile)

	DEN = av_mol_den

	P_z_theta_phi_varphi = np.zeros((nslice,npi,npi,npi))
	rho_z_theta = np.zeros((nslice,npi))
	rho_z_phi = np.zeros((nslice,npi))
	rho_z_varphi = np.zeros((nslice,npi))
	mol_count = np.zeros(nslice)

	int_P_z_theta_phi_varphi1 = np.zeros((nslice,npi,npi,npi))
	int_rho_z_theta1 = np.zeros((nslice,npi))
	int_rho_z_phi1 = np.zeros((nslice,npi))
	int_rho_z_varphi1 = np.zeros((nslice,npi))
	int_mol_count1 = np.zeros(nslice)

	int_P_z_theta_phi_varphi2 = np.zeros((nslice,npi,npi,npi))
	int_rho_z_theta2 = np.zeros((nslice,npi))
	int_rho_z_phi2 = np.zeros((nslice,npi))
	int_rho_z_varphi2 = np.zeros((nslice,npi))
	int_mol_count2 = np.zeros(nslice)

	z_array = np.zeros((nimage, nmol))
	theta = np.zeros((nimage, nmol))
	phi = np.zeros((nimage, nmol))
	varphi = np.zeros((nimage, nmol))

	zeta_array1 = np.zeros((nimage, nmol))
	int_theta1 = np.zeros((nimage, nmol))
	int_phi1 = np.zeros((nimage, nmol))
	int_varphi1 = np.zeros((nimage, nmol))

	zeta_array2 = np.zeros((nimage, nmol))
	int_theta2 = np.zeros((nimage, nmol))
	int_phi2 = np.zeros((nimage, nmol))
	int_varphi2 = np.zeros((nimage, nmol))

	""" Cycle through existing ANGLE files""" 
	start_image_P = 0

	for image in xrange(nimage):
		sys.stdout.write("CHECKING {} out of {} ANGLE files\r".format(image+1, nimage) )
		sys.stdout.flush()
		if os.path.exists('{}/DATA/EULER/{}_{}_{}_ANGLE.txt'.format(root, model.lower(), csize, image)) and ow_P.upper() != 'Y':
			start_image_P = image + 1
			with file('{}/DATA/EULER/{}_{}_{}_ANGLE.txt'.format(root, model.lower(), csize, image), 'r') as infile:
				z_array[image], theta[image], phi[image], varphi[image] = np.loadtxt(infile)
			with file('{}/DATA/INTEULER/{}_{}_{}_{}_ANGLE1.txt'.format(root, model.lower(), csize, nm, image), 'r') as infile:
				zeta_array1[image], int_theta1[image], int_phi1[image], int_varphi1[image] = np.loadtxt(infile)
			with file('{}/DATA/INTEULER/{}_{}_{}_{}_ANGLE2.txt'.format(root, model.lower(), csize, nm, image), 'r') as infile:
				zeta_array2[image], int_theta2[image], int_phi2[image], int_varphi2[image] = np.loadtxt(infile)
		sys.stdout.write(" "*80 + "\r")
	print "FOUND {} out of {} ANGLE files\n".format(start_image_P, nimage)

	for image in xrange(start_image_P, nimage):
		sys.stdout.write("PROCESSING {} out of {} ANGLE files\r".format(image+1, nimage) )
		sys.stdout.flush()

		with file('{}/DATA/ACOEFF/{}_{}_{}_{}_INTCOEFF.txt'.format(root, model.lower(), csize, nm, image), 'r') as infile:
			auv1, auv2 = np.loadtxt(infile)

		ZYX = np.rot90(traj.xyz[image])
		zat = ZYX[0] * 10
		yat = ZYX[1] * 10
		xat = ZYX[2] * 10

		#xat, yat, zat = ut.read_positions("{}/{}_{}_{}{}.rst".format(root,model.lower(), csize, suffix, image), nsite)
		xmol, ymol, zmol = ut.molecules(xat, yat, zat, nsite, M, com)
		xR, yR, zR = ut.centre_mass(xat, yat, zat, nsite, M)

		for j in xrange(nmol):
			for l in xrange(nsite):
				MOLECULES[image][j][l][0] = xat[j*nsite+l]
				MOLECULES[image][j][l][1] = yat[j*nsite+l]
				MOLECULES[image][j][l][2] = zat[j*nsite+l]

			zeta1 = zmol[j] - zR - IS.xi(xmol[j], ymol[j], nm, auv1, DIM) 
			dzx1, dzy1 = IS.dxyi(xmol[j], ymol[j], nm, auv1, DIM)
			
			zeta2 = - zmol[j] + zR + IS.xi(xmol[j], ymol[j], nm, auv2, DIM)			
			dzx2, dzy2 = IS.dxyi(xmol[j], ymol[j], nm, auv2, DIM)

			T = local_frame_molecule(MOLECULES[image][j], model) 
			z_array[image][j] = zmol[j] - zR
			theta[image][j] = np.arccos(T[2][2])
			phi[image][j] = np.arctan2(-T[2][0],T[2][1])
			varphi[image][j] = np.arctan2(T[0][2],T[1][2])
	
			
			zeta_array1[image][j] = zeta1
			int_theta1[image][j] = np.arccos(T[2][2])
			int_phi1[image][j] = np.arctan2(-T[2][0],T[2][1])
			int_varphi1[image][j] = np.arctan2(T[0][2],T[1][2])

			zeta_array2[image][j] = zeta2
			int_theta2[image][j] = np.arccos(T[2][2])
			int_phi2[image][j] = np.arctan2(-T[2][0],T[2][1])
			int_varphi2[image][j] = np.arctan2(T[0][2],T[1][2])

			"""
			O = ut.local_frame_surface(dzx1, dzy1, zeta1, zR[image])
			T1 = np.dot(T, np.linalg.inv(O))
			if T1[2][2] < -1: T1[2][2] = -1.0
			elif T1[2][2] > 1: T1[2][2] = 1.0
			zeta_array1[image][j] = zeta1
			int_theta1[image][j] = np.arccos(T1[2][2])
			int_phi1[image][j] = np.arctan2(-T1[2][0],T1[2][1])
			int_varphi1[image][j] = np.arctan2(T1[0][2],T1[1][2])

			O = ut.local_frame_surface(dzx2, dzy2, zeta2, zR[image])
			T2 = np.dot(T, np.linalg.inv(O))
			if T2[2][2] < -1: T2[2][2] = -1.0
			elif T2[2][2] > 1: T2[2][2] = 1.0
			zeta_array2[image][j] = zeta2
			int_theta2[image][j] = np.arccos(T2[2][2])
			int_phi2[image][j] = np.arctan2(-T2[2][0],T2[2][1])
			int_varphi2[image][j] = np.arctan2(T2[0][2],T2[1][2])
			"""

		with file('{}/DATA/EULER/{}_{}_{}_ANGLE.txt'.format(root, model.lower(), csize, image), 'w') as outfile:
			np.savetxt(outfile, (z_array[image], theta[image], phi[image], varphi[image]), fmt='%-12.6f')
		with file('{}/DATA/INTEULER/{}_{}_{}_{}_ANGLE1.txt'.format(root, model.lower(), csize,  nm, image), 'w') as outfile:
			np.savetxt(outfile, (zeta_array1[image], int_theta1[image], int_phi1[image], int_varphi1[image]), fmt='%-12.6f')
		with file('{}/DATA/INTEULER/{}_{}_{}_{}_ANGLE2.txt'.format(root, model.lower(), csize, nm, image), 'w') as outfile:
			np.savetxt(outfile, (zeta_array2[image], int_theta2[image], int_phi2[image], int_varphi2[image]), fmt='%-12.6f')

	for i in xrange(nimage):
		for j in xrange(nmol):
			
			z = z_array[i][j] + DIM[2]/2.
			index1 = int(z * nslice / DIM[2]) % nslice
			index2 = int(theta[i][j] * npi / np.pi) % npi
			index3 = int(phi[i][j] * npi / np.pi) % npi
			index4 = int(varphi[i][j] * npi / np.pi) % npi

			P_z_theta_phi_varphi[index1][index2][index3][index4] += 1
			rho_z_theta[index1][index2] += 1
			rho_z_phi[index1][index3] += 1
			rho_z_varphi[index1][index4] += 1
			mol_count[index1] += 1
	
			z = zeta_array1[i][j] + DIM[2]/2.
			index1 = int(z * nslice / DIM[2]) % nslice
			index2 = int(int_theta1[i][j] * npi / np.pi) % npi
			index3 = int(int_phi1[i][j] * npi / np.pi) % npi
			index4 = int(int_varphi1[i][j] * npi / np.pi) % npi

			int_P_z_theta_phi_varphi1[index1][index2][index3][index4] += 1
			int_rho_z_theta1[index1][index2] += 1
			int_rho_z_phi1[index1][index3] += 1
			int_rho_z_varphi1[index1][index4] += 1
			int_mol_count1[index1] += 1

			z = zeta_array2[i][j] + DIM[2]/2.
			index1 = int(z * nslice / DIM[2]) % nslice
			index2 = int(int_theta2[i][j] * npi / np.pi) % npi
			index3 = int(int_phi2[i][j] * npi / np.pi) % npi
			index4 = int(int_varphi2[i][j] * npi / np.pi) % npi
	
			int_P_z_theta_phi_varphi2[index1][index2][index3][index4] += 1
			int_rho_z_theta2[index1][index2] += 1
			int_rho_z_phi2[index1][index3] += 1
			int_rho_z_varphi2[index1][index4] += 1
			int_mol_count2[index1] += 1

	int_P_z_theta_phi_varphi = (int_P_z_theta_phi_varphi1 + int_P_z_theta_phi_varphi2) / 2.
	int_rho_z_theta = (int_rho_z_theta1 + int_rho_z_theta2) / 2.
	int_rho_z_phi = (int_rho_z_phi1 + int_rho_z_phi2) / 2.
	int_rho_z_varphi = (int_rho_z_varphi1 + int_rho_z_varphi2) / 2.
	int_mol_count = (int_mol_count1 + int_mol_count2) / 2.

	for index1 in xrange(nslice): 
		if mol_count[index1] != 0:
			P_z_theta_phi_varphi[index1] = P_z_theta_phi_varphi[index1] / mol_count[index1]
			rho_z_theta[index1] = rho_z_theta[index1] / mol_count[index1]
			rho_z_phi[index1] = rho_z_phi[index1] / mol_count[index1]
			rho_z_varphi[index1] = rho_z_varphi[index1] / mol_count[index1]

		if int_mol_count[index1] != 0:
			int_P_z_theta_phi_varphi[index1] = int_P_z_theta_phi_varphi[index1] / int_mol_count[index1]
			int_rho_z_theta[index1] = int_rho_z_theta[index1] / int_mol_count[index1]
			int_rho_z_phi[index1] = int_rho_z_phi[index1] / int_mol_count[index1]
			int_rho_z_varphi[index1] = int_rho_z_varphi[index1] / int_mol_count[index1]

		if int_mol_count1[index1] != 0:
			int_P_z_theta_phi_varphi1[index1] = int_P_z_theta_phi_varphi1[index1] / int_mol_count1[index1]
			int_rho_z_theta1[index1] = int_rho_z_theta1[index1] / int_mol_count1[index1]
			int_rho_z_phi1[index1] = int_rho_z_phi1[index1] / int_mol_count1[index1]
			int_rho_z_varphi1[index1] = int_rho_z_varphi1[index1] / int_mol_count1[index1]

		if int_mol_count2[index1] != 0:
			int_P_z_theta_phi_varphi2[index1] = int_P_z_theta_phi_varphi2[index1] / int_mol_count2[index1]
			int_rho_z_theta2[index1] = int_rho_z_theta2[index1] / int_mol_count2[index1]
			int_rho_z_phi2[index1] = int_rho_z_phi2[index1] / int_mol_count2[index1]
			int_rho_z_varphi2[index1] = int_rho_z_varphi2[index1] / int_mol_count2[index1]

	av_theta = np.zeros(nslice)
	av_phi = np.zeros(nslice)
	av_varphi = np.zeros(nslice)
	P1 = np.zeros(nslice)
	P2 = np.zeros(nslice)

	int_av_theta = np.zeros(nslice)
	int_av_phi = np.zeros(nslice)
	int_av_varphi = np.zeros(nslice)
	int_P1 = np.zeros(nslice)
	int_P2 = np.zeros(nslice)

	int_av_theta1 = np.zeros(nslice)
	int_av_phi1 = np.zeros(nslice)
	int_av_varphi1 = np.zeros(nslice)
	int_P11 = np.zeros(nslice)
	int_P21 = np.zeros(nslice)

	int_av_theta2 = np.zeros(nslice)
	int_av_phi2 = np.zeros(nslice)
	int_av_varphi2 = np.zeros(nslice)
	int_P12 = np.zeros(nslice)
	int_P22 = np.zeros(nslice)


	X_trig = np.linspace(0, np.pi, npi)
	X_trig1 = np.linspace(0, np.pi/2, npi)
	X_trig2 = np.linspace(0, np.pi/4, npi)

	for index1 in xrange(nslice): 
		for index2 in xrange(npi):
			av_theta[index1] += rho_z_theta[index1][index2] * X_trig[index2] # * 180./np.pi
			av_phi[index1] += rho_z_phi[index1][index2] * X_trig[index2]  # * 180./np.pi
			av_varphi[index1] += rho_z_varphi[index1][index2] * X_trig[index2]  # * 180./np.pi
			P1[index1] += rho_z_theta[index1][index2] * np.cos(X_trig[index2])
			P2[index1] += rho_z_theta[index1][index2] * 0.5 * (3 * np.cos(X_trig[index2])**2 - 1)

			int_av_theta[index1] += int_rho_z_theta[index1][index2] * X_trig[index2] # * 180./np.pi
			int_av_phi[index1] += int_rho_z_phi[index1][index2] * X_trig[index2]  # * 180./np.pi
			int_av_varphi[index1] += int_rho_z_varphi[index1][index2] * X_trig[index2]  # * 180./np.pi
			int_P1[index1] += int_rho_z_theta[index1][index2] * np.cos(X_trig[index2])
			int_P2[index1] += int_rho_z_theta[index1][index2] * 0.5 * (3 * np.cos(X_trig[index2])**2 - 1)

			int_av_theta1[index1] += int_rho_z_theta1[index1][index2] * X_trig[index2] # * 180./np.pi
			int_av_phi1[index1] += int_rho_z_phi1[index1][index2] * X_trig[index2]  # * 180./np.pi
			int_av_varphi1[index1] += int_rho_z_varphi1[index1][index2] * X_trig[index2]  # * 180./np.pi
			int_P11[index1] += int_rho_z_theta1[index1][index2] * np.cos(X_trig[index2])
			int_P21[index1] += int_rho_z_theta1[index1][index2] * 0.5 * (3 * np.cos(X_trig[index2])**2 - 1)

			int_av_theta2[index1] += int_rho_z_theta2[index1][index2] * X_trig[index2] # * 180./np.pi
			int_av_phi2[index1] += int_rho_z_phi2[index1][index2] * X_trig[index2]  # * 180./np.pi
			int_av_varphi2[index1] += int_rho_z_varphi2[index1][index2] * X_trig[index2]  # * 180./np.pi
			int_P12[index1] += int_rho_z_theta2[index1][index2] * np.cos(X_trig[index2])
			int_P22[index1] += int_rho_z_theta2[index1][index2] * 0.5 * (3 * np.cos(X_trig[index2])**2 - 1)


	water_au_A = 0.5291772083

	water_exp_a = [1.528, 1.415, 1.468]
	water_CCSD_a_1 = np.array([10.18, 9.72, 9.87]) * 0.529**3
	water_CCSD_a_2 = np.array([10.11, 9.59, 9.78]) * 0.529**3
	water_CCSD_a_3 = np.array([10.09, 9.55, 9.75]) * 0.529**3
	water_CCSD_a_4 = np.array([9.98, 9.35, 9.61]) * 0.529**3
	water_ame_a = [1.672, 1.225, 1.328]
	water_abi_a = [1.47, 1.38, 1.42]
	water_tip_a = [0, 2.55, 0.82]

	water_exp_w = 514.5
	water_a0 = [10.683, 10.408, 10.534]
	water_S4 = [31.91, 66.22, 43.31]

	methanol_exp_a = [3.524, 3.091, 3.012]

	c = 514.5 / 0.08856

	w = 1000 / c

	#a = map (lambda i: (a0[i] + S4[i] * w**2) * au_A**3, range(3))
	
	if model.upper() == 'METHANOL':
		if a_type == 'exp': a = methanol_exp_a
	else:
		if a_type == 'exp': a = water_exp_a
		elif a_type == 'ame': a = water_ame_a
		elif a_type == 'abi': a = water_abi_a

	axx = np.zeros(nslice)
	azz = np.zeros(nslice)
	q1 = np.zeros(nslice)
	q2 = np.zeros(nslice)

	int_axx = np.zeros(nslice)
	int_azz = np.zeros(nslice)
	int_q1 = np.zeros(nslice)
	int_q2 = np.zeros(nslice)

	int_axx1 = np.zeros(nslice)
	int_azz1 = np.zeros(nslice)
	int_q11 = np.zeros(nslice)
	int_q21 = np.zeros(nslice)

	int_axx2 = np.zeros(nslice)
	int_azz2 = np.zeros(nslice)
	int_q12 = np.zeros(nslice)
	int_q22 = np.zeros(nslice)


	vd = DIM[0] * DIM[1] * DIM[2] / nslice

	print "PROCESSING POLARISABILITIES"
	for n in xrange(nslice):
		for i in xrange(npi):
			for j in xrange(npi):

				axx[n] += (a[0] * (np.cos(X_trig[i])**2 * np.cos(X_trig[j])**2 + np.sin(X_trig[j])**2) 
				+ a[1] * (np.cos(X_trig[i])**2 * np.sin(X_trig[j])**2 + np.cos(X_trig[j])**2) 
				+ a[2] * np.sin(X_trig[i])**2) * np.sum(P_z_theta_phi_varphi[n][i][j]) * 0.5

				azz[n] += (a[0] * np.sin(X_trig[i])**2 * np.cos(X_trig[j])**2 
				+ a[1] * np.sin(X_trig[i])**2 * np.sin(X_trig[j])**2 
				+ a[2] * np.cos(X_trig[i])**2) * np.sum(P_z_theta_phi_varphi[n][i][j])

				int_axx[n] += (a[0] * (np.cos(X_trig[i])**2 * np.cos(X_trig[j])**2 + np.sin(X_trig[j])**2) 
				+ a[1] * (np.cos(X_trig[i])**2 * np.sin(X_trig[j])**2 + np.cos(X_trig[j])**2) 
				+ a[2] * np.sin(X_trig[i])**2) * np.sum(int_P_z_theta_phi_varphi[n][i][j]) * 0.5

				int_azz[n] += (a[0] * np.sin(X_trig[i])**2 * np.cos(X_trig[j])**2 
				+ a[1] * np.sin(X_trig[i])**2 * np.sin(X_trig[j])**2 
				+ a[2] * np.cos(X_trig[i])**2) * np.sum(int_P_z_theta_phi_varphi[n][i][j])

				int_axx1[n] += (a[0] * (np.cos(X_trig[i])**2 * np.cos(X_trig[j])**2 + np.sin(X_trig[j])**2) 
				+ a[1] * (np.cos(X_trig[i])**2 * np.sin(X_trig[j])**2 + np.cos(X_trig[j])**2) 
				+ a[2] * np.sin(X_trig[i])**2) * np.sum(int_P_z_theta_phi_varphi1[n][i][j]) * 0.5

				int_azz1[n] += (a[0] * np.sin(X_trig[i])**2 * np.cos(X_trig[j])**2 
				+ a[1] * np.sin(X_trig[i])**2 * np.sin(X_trig[j])**2 
				+ a[2] * np.cos(X_trig[i])**2) * np.sum(int_P_z_theta_phi_varphi1[n][i][j])

				int_axx2[n] += (a[0] * (np.cos(X_trig[i])**2 * np.cos(X_trig[j])**2 + np.sin(X_trig[j])**2) 
				+ a[1] * (np.cos(X_trig[i])**2 * np.sin(X_trig[j])**2 + np.cos(X_trig[j])**2) 
				+ a[2] * np.sin(X_trig[i])**2) * np.sum(int_P_z_theta_phi_varphi2[n][i][j]) * 0.5

				int_azz2[n] += (a[0] * np.sin(X_trig[i])**2 * np.cos(X_trig[j])**2 
				+ a[1] * np.sin(X_trig[i])**2 * np.sin(X_trig[j])**2 
				+ a[2] * np.cos(X_trig[i])**2) * np.sum(int_P_z_theta_phi_varphi2[n][i][j])


			q1[n] += (3 * np.cos(X_trig[i])**2 - 1) * rho_z_theta[n][i] * DEN[n] * 0.5 * np.pi / npi
			q2[n] += P2[n] * DEN[n] * np.pi / npi

			int_q1[n] += (3 * np.cos(X_trig[i])**2 - 1) * int_rho_z_theta[n][i] * int_av_mol_den[n] * 0.5 * np.pi / npi
			int_q2[n] += int_P2[n] * int_av_mol_den[n] * np.pi / npi

			int_q11[n] += (3 * np.cos(X_trig[i])**2 - 1) * int_rho_z_theta1[n][i] * w_den_1[n] * 0.5 * np.pi / npi
			int_q21[n] += int_P21[n] * w_den_1[n] * np.pi / npi

			int_q12[n] += (3 * np.cos(X_trig[i])**2 - 1) * int_rho_z_theta2[n][i] * w_den_2[n] * 0.5 * np.pi / npi
			int_q22[n] += int_P22[n] * w_den_2[n] * np.pi / npi

	plt.plot(range(len(int_axx)), axx)
	plt.plot(range(len(int_axx)), int_axx)
	plt.show()


	print ""
	print "WRITING TO FILE..."

	with file('{}/DATA/EULER/{}_{}_{}_{}_{}_EUL.txt'.format(root, model.lower(), csize, nslice, a_type, nimage), 'w') as outfile:
		np.savetxt(outfile, (axx,azz,q1,q2,av_theta,av_phi,av_varphi,P1,P2), fmt='%-12.6f')
	with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_EUL.txt'.format(root, model.lower(), csize, nslice, a_type, nm, nimage), 'w') as outfile:
		np.savetxt(outfile, (int_axx,int_azz,int_q1,int_q2,int_av_theta,int_av_phi,int_av_varphi,int_P1,int_P2), fmt='%-12.6f')
	with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_EUL1.txt'.format(root, model.lower(), csize, nslice, a_type, nm, nimage), 'w') as outfile:
		np.savetxt(outfile, (int_axx1,int_azz1,int_q11,int_q21,int_av_theta1,int_av_phi1,int_av_varphi1,int_P11,int_P21), fmt='%-12.6f')
	with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_EUL2.txt'.format(root, model.lower(), csize, nslice, a_type, nm, nimage), 'w') as outfile:
		np.savetxt(outfile, (int_axx2,int_azz2,int_q12,int_q22,int_av_theta2,int_av_phi2,int_av_varphi2,int_P12,int_P22), fmt='%-12.6f')
							
	print "{} {} {} COMPLETE\n".format(root, model.upper(), csize)


def local_frame_molecule(molecule, model):
	
	if model.upper() == 'METHANOL': 
		d = np.subtract(molecule[2], molecule[3])
		t = np.subtract(molecule[4], molecule[1])
	else: 
		d = np.add(np.subtract(molecule[0], molecule[1]), np.subtract(molecule[0],molecule[2]))
		t = np.subtract(molecule[1], molecule[2])

	d = ut.unit_vector(d[0], d[1], d[2])
	t = ut.unit_vector(t[0], t[1], t[2])
	n = np.cross(d, t)
	
	B = np.array([[n[0], t[0], d[0]], [n[1], t[1], d[1]], [n[2], t[2], d[2]]])
	
	return B


def main(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, TYPE, folder, nfolder, suffix, ntraj):

	for i in xrange(nfolder):
		directory = '{}/{}_{}'.format(root, TYPE.upper(), i)
		if not os.path.exists('{}/{}/{}_{}_{}_{}.nc'.format(directory, folder.upper(), model.lower(), csize, suffix, 800)): 
			ut.make_nc(directory, folder.upper(),  model.lower(), csize, suffix, ntraj, 'N')
		traj = ut.load_nc(directory, folder.upper())							
		directory = '{}/{}'.format(directory, folder.upper())

		natom = traj.n_atoms
		nmol = traj.n_residues
		DIM = np.array(traj.unitcell_lengths[0]) * 10
		sigma = np.max(LJ[1])
		lslice = 0.05 * sigma
		nslice = int(DIM[2] / lslice)
		vlim = 3
		ncube = 3
		nm = int(DIM[0] / sigma)
		nxy = 30

		if not os.path.exists("{}/DATA/EULER".format(root)): os.mkdir("{}/DATA/EULER".format(root))
		if not os.path.exists("{}/DATA/INTEULER".format(root)): os.mkdir("{}/DATA/INTEULER".format(root))

		if os.path.exists('{}/DATA/EULER/{}_{}_{}_{}_{}_EUL.txt'.format(root, model.lower(), csize, nslice, a_type, nimage)):
			print '\nFILE FOUND {}/DATA/EULER/{}_{}_{}_{}_{}_EUL.txt'.format(root, model.lower(), csize, nslice, a_type, nimage)
			overwrite = raw_input("OVERWRITE? (Y/N): ")
			if overwrite.upper() == 'Y':  
				ow_angles = raw_input("OVERWRITE ANGLES? (Y/N): ") 
				euler_profile(traj, root, nimage, nslice, nmol, model, csize, suffix, DIM, nsite, a_type, nm, nxy, ow_angles)
		else: euler_profile(traj, root, nimage, nslice, nmol, model, csize, suffix, DIM, nsite, a_type, nm, nxy, 'N')


