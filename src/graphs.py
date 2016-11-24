"""
*************** POSITION ANALYSIS *******************

PROGRAM INPUT:
	       

PROGRAM OUTPUT: 
		
		

***************************************************************
Created 11/05/16 by Frank Longford

Last modified 11/05/16 by Frank Longford
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


def print_graphs(root, groot, model, csize, nslice, nimage, a_type, force, nm, DIM):

	print "PRINTING GRAPHS..."

	nsite, AT, Q, M, LJ = ut.get_param(model)
	nH = 2

	with file('{}/DATA/DEN/{}_{}_{}_{}_PAR.txt'.format(root, model.lower(), csize, nslice, nimage), 'r') as infile:
		param = np.loadtxt(infile)

	with file('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, nimage), 'r') as infile:
		av_mass_den, av_atom_den, av_mol_den, av_H_den = np.loadtxt(infile)

	with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, nm, nimage), 'r') as infile:
		int_av_mass_den, int_av_atom_den, int_av_mol_den, int_av_H_den, w_den_1, w_den_2 = np.loadtxt(infile)

	with file('{}/DATA/EULER/{}_{}_{}_{}_{}_EUL.txt'.format(root, model.lower(), csize, nslice, a_type, nimage), 'r') as infile:
		axx, azz, q1, q2, av_theta, av_phi, av_varphi, P1, P2 = np.loadtxt(infile)

	with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_EUL.txt'.format(root, model.lower(), csize, nslice, a_type, nm, nimage), 'r') as infile:
		int_axx, int_azz, int_q1, int_q2, int_av_theta, int_av_phi, int_av_varphi, int_P1, int_P2 = np.loadtxt(infile)

	with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_EUL1.txt'.format(root, model.lower(), csize, nslice, a_type, nm, nimage), 'r') as infile:
		int_axx1, int_azz1, int_q11, int_q21, int_av_theta1, int_av_phi1, int_av_varphi1, int_P11, int_P21 = np.loadtxt(infile)

	with file('{}/DATA/INTEULER/{}_{}_{}_{}_{}_{}_EUL2.txt'.format(root, model.lower(), csize, nslice, a_type, nm, nimage), 'r') as infile:
		int_axx2, int_azz2, int_q12, int_q22, int_av_theta2, int_av_phi2, int_av_varphi2, int_P12, int_P22 = np.loadtxt(infile)

	with file('{}/DATA/DIELEC/{}_{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, a_type, nimage), 'r') as infile:
		exx, ezz, no, ne, ni = np.loadtxt(infile)
	
	with file('{}/DATA/DIELEC/{}_{}_{}_{}_{}_{}_INTDEN.txt'.format(root, model.lower(), csize, nslice, a_type, nm, nimage), 'r') as infile:
		int_exx, int_ezz, int_no, int_ne, int_ni = np.loadtxt(infile)

	with file('{}/DATA/DIELEC/{}_{}_{}_{}_{}_{}_WINTDEN.txt'.format(root, model.lower(), csize, nslice, a_type, nm, nimage), 'r') as infile:
		int_exx1, int_ezz1, int_no1, int_ne1, int_ni1, int_exx2, int_ezz2, int_no2, int_ne2, int_ni2 = np.loadtxt(infile)

	Z = np.linspace(-DIM[2]/2., DIM[2]/2., nslice)
	Z_surf = np.linspace(- DIM[2]/2.+param[3], DIM[2]/2.+param[3], nslice)
	int_Z = np.linspace(- DIM[2]/2.-param[3], DIM[2]/2.-param[3], nslice)
	#print np.max(int_DEN), int_DEN[int(nslice/2)], av_mass_den[int(nslice/2)], av_mass_den[int(nslice/2)]/av_mol_den[int(nslice/2)] * np.max(int_DEN), param[0], param[0] * av_mol_den[int(nslice/2)]/av_mass_den[int(nslice/2)]

	#aI = map (lambda n: (azz[n] + 2*axx[n]) / 3, range(nslice))
	#aA = map (lambda n: (azz[n] - axx[n]) / 3, range(nslice))

	#eI = map (lambda n: (ezz[n] + 2*exx[n]) / 3, range(nslice))
	#eA = map (lambda n: (ezz[n] - exx[n]) / 3, range(nslice))

	#nI = map (lambda n: np.sqrt(ur * eI[n]), range(nslice))
	#nA = map (lambda n: np.sqrt(ur * eA[n]), range(nslice))

	fig_x = 14
	fig_y = 9

	plt.rc('text', usetex=True)
	font = {'family' : 'normal','weight' : 'bold', 'size'   : 22}
	plt.rc('font', **font)
	plt.rc('xtick', labelsize=20)
	plt.rc('ytick', labelsize=20)

	i = 0
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\rho(z)$ (g cm$^{-3}$)')
	plt.scatter(Z, av_mass_den, color='r')
	plt.plot(Z, av_mass_den, color='r')
	plt.axis([Z[0],Z[-1],0,1.1])
	plt.savefig('{}/{}_{}_{}_av_mass_den.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\rho(z)$ (\AA$^{-3}$)')
	plt.scatter(Z, av_atom_den, color='b')
	plt.plot(Z, av_atom_den, color='b')
	plt.axis([Z[0],Z[-1],np.min((av_atom_den)),np.max((av_atom_den))+0.1*np.mean(av_atom_den)])
	plt.savefig('{}/{}_{}_{}_av_atom_den.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\rho(z)$ ( \AA$^{-3}$)')
	plt.scatter(Z, av_mol_den, color='g')
	plt.plot(Z, av_mol_den, color='g')
	plt.scatter(Z, av_H_den/nH, color='black')
	plt.plot(Z, av_H_den/nH, color='black')
	plt.axis([Z[0],Z[-1],0,0.05])
	plt.savefig('{}/{}_{}_{}_av_mol_den.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\rho(z)$ ( \AA$^{-3}$)')
	plt.scatter(Z_surf, av_mol_den, color='g')
	plt.plot(Z_surf, av_mol_den, color='g')
	plt.scatter(Z_surf, av_H_den/nH, color='black')
	plt.plot(Z_surf, av_H_den/nH, color='black')
	plt.axis([-10,+10,0,0.1])
	plt.savefig('{}/{}_{}_{}_av_mol_den_surf.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z, axx, label=r'$\bar{\alpha}_{\parallel}$', color='r', marker='o')
	plt.scatter(Z, azz, label=r'$\bar{\alpha}_{\perp}$', color='b', marker='o')
	plt.axis([Z[0],Z[-1],1.43,1.52])
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\bar{a}$ (\AA$^3$)')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_polarisability.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z_surf, axx, label=r'$\bar{\alpha}_{\parallel}$', color='r', marker='o')
	plt.scatter(Z_surf, azz, label=r'$\bar{\alpha}_{\perp}$', color='b', marker='o')
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\bar{a}$ (\AA$^3$)')
	plt.legend(loc=3)
	plt.axis([-10,+10,1.43,1.52])
	plt.savefig('{}/{}_{}_{}_polarisability_surf.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z, exx, label=r'$\epsilon_{r_\parallel}$', color='r', marker='o')
	plt.scatter(Z, ezz, label=r'$\epsilon_{r_\perp}$', color='b', marker='x')
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\epsilon$ (a.u.)')
	plt.legend(loc=3)
	plt.axis([Z[0],Z[-1],1.00,2.1])
	plt.savefig('{}/{}_{}_{}_dielectric.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z_surf, exx, label=r'$\epsilon_{r_\parallel}$', color='r', marker='o')
	plt.scatter(Z_surf, ezz, label=r'$\epsilon_{r_\perp}$', color='b', marker='x')
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\epsilon$ (a.u.)')
	plt.legend(loc=3)
	plt.axis([-10,+10,1.00,4.7])
	plt.savefig('{}/{}_{}_{}_dielectric_surf.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	#plt.scatter(Z, ni, label=r'$n_{i}$', color='r', marker='o')
	plt.scatter(Z, no, label=r'$n_{o}$', color='r', marker='o')
	plt.scatter(Z, ne, label=r'$n_{e}$', color='b', marker='x')
	plt.axis([Z[0],Z[-1],1,1.5])
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'n (a.u.)')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_refractive_index.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	#plt.scatter(Z, ni, label=r'$n_{i}$', color='r', marker='o')
	plt.scatter(Z_surf, no, label=r'$n_{o}$', color='r', marker='o')
	plt.scatter(Z_surf, ne, label=r'$n_{e}$', color='b', marker='x')
	plt.axis([-10,+10,1,2.2])
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'n (a.u.)')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_refractive_index_surf.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z, av_theta, color='r')
	#plt.scatter(Z, av_phi, color='b')
	#plt.scatter(Z, av_varphi, color='g')
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\theta$ (radians)')
	plt.savefig('{}/{}_{}_{}_angles.pdf'.format(groot,nimage,nslice,force))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z, P1, color='r')
	#plt.scatter(Z, P2, color='b')
	plt.xlabel(r'z Coordinate (\AA)')
	#plt.ylabel(r'Z (a.u.)')
	plt.savefig('{}/{}_{}_{}_orientation.pdf'.format(groot,nimage,nslice,force))
	

	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\rho(z)$ (g cm$^{-3}$)')
	plt.scatter(Z, int_av_mass_den, color='r')
	plt.plot(Z, int_av_mass_den, color='r')
	plt.axis([-10,+10,0,3.1])
	plt.savefig('{}/{}_{}_{}_{}_int_av_mass_den.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\rho(z)$ (\AA$^{-3}$)')
	plt.scatter(Z, int_av_atom_den, color='b')
	plt.plot(Z, int_av_atom_den, color='b')
	plt.axis([-10,+10,np.min((int_av_atom_den)),np.max((int_av_atom_den))+0.1*np.mean(int_av_atom_den)])
	plt.savefig('{}/{}_{}_{}_{}_int_av_atom_den.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\rho(z)$ ( \AA$^{-3}$)')
	plt.scatter(Z, int_av_mol_den, color='g')
	plt.scatter(Z, int_av_H_den/nH, color='black')
	plt.plot(Z, int_av_mol_den, color='g')
	plt.plot(Z, int_av_H_den/nH, color='black')
	plt.axis([-10,+10,0,0.1])
	plt.savefig('{}/{}_{}_{}_{}_int_av_mol_den.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\rho(z)$ ( \AA$^{-3}$)')
	#plt.scatter(int_Z, w_den_1, color='red')
	plt.plot(Z, w_den_1, color='red')
	#plt.scatter(int_Z, w_den_2, color='b')
	plt.plot(Z, w_den_2, color='b')
	plt.axis([Z[0],Z[-1],0.0,0.1])
	plt.savefig('{}/{}_{}_{}_{}_int_DEN.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	X1 = [int_axx1[np.mod(i+int(param[3]/DIM[2] * nslice), nslice)] for i in xrange(nslice)]
	X2 = [int_azz1[np.mod(i+int(param[3]/DIM[2] * nslice), nslice)] for i in xrange(nslice)]
	X3 = [int_axx2[np.mod(i+int(param[3]/DIM[2] * nslice), nslice)] for i in xrange(nslice)][::-1]
	X4 = [int_azz2[np.mod(i+int(param[3]/DIM[2] * nslice), nslice)] for i in xrange(nslice)][::-1]
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z, X1, label=r'$\bar{\alpha}_{\parallel}$', color='r', marker='o')
	plt.scatter(Z, X2, label=r'$\bar{\alpha}_{\perp}$', color='b', marker='o')
	plt.scatter(Z, X3, label=r'$\bar{\alpha}_{\parallel}$', color='g', marker='o')
	plt.scatter(Z, X4, label=r'$\bar{\alpha}_{\perp}$', color='black', marker='x')
	plt.axis([Z[0],Z[-1],1.43,1.52])
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\bar{a}$ (\AA$^3$)')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_{}_int_polarisability.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z, int_axx, label=r'$\bar{\alpha}_{\parallel}$', color='r', marker='o')
	plt.scatter(Z, int_azz, label=r'$\bar{\alpha}_{\perp}$', color='b', marker='o')
	plt.axis([-10,+10,1.43,1.52])
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\bar{a}$ (\AA$^3$)')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_{}_int_polarisability_surf.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z, int_exx1, label=r'$\epsilon_{r_\parallel}$', color='r', marker='o')
	plt.scatter(Z, int_ezz1, label=r'$\epsilon_{r_\perp}$', color='b', marker='x')
	plt.scatter(Z, int_exx2, label=r'$\epsilon_{r_\parallel}$', color='g', marker='o')
	plt.scatter(Z, int_ezz2, label=r'$\epsilon_{r_\perp}$', color='black', marker='x')
	plt.axis([Z[0],Z[-1],1.00,2.1])
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\epsilon$ (a.u.)')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_{}_int_dielectric.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z, int_exx, label=r'$\epsilon_{r_\parallel}$', color='r', marker='o')
	plt.scatter(Z, int_ezz, label=r'$\epsilon_{r_\perp}$', color='b', marker='x')
	plt.axis([-10,+10,1,4.7])
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\epsilon$ (a.u.)')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_{}_int_dielectric_surf.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z, int_no1, label=r'$n_{o}$', color='r', marker='o')
	plt.scatter(Z, int_ne1, label=r'$n_{e}$', color='b', marker='x')
	plt.scatter(Z, int_no2, label=r'$n_{o}$', color='g', marker='o')
	plt.scatter(Z, int_ne2, label=r'$n_{e}$', color='black', marker='x')
	plt.axis([Z[0],Z[-1],1,1.5])
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'n (a.u.)')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_{}_int_refractive_index.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z, int_no, label=r'$n_{o}$', color='r', marker='o')
	plt.scatter(Z, int_ne, label=r'$n_{e}$', color='b', marker='x')
	plt.axis([-10,+10,1,2.2])
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'n (a.u.)')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_{}_int_refractive_index_surf.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z_surf, int_av_theta, color='r')
	#plt.scatter(Z, av_phi, color='b')
	#plt.scatter(Z, av_varphi, color='g')
	plt.xlabel(r'z Coordinate (\AA)')
	plt.ylabel(r'$\theta$ (radians)')
	plt.savefig('{}/{}_{}_{}_{}_int_angles.pdf'.format(groot,nimage,nslice,force,nm))
	i += 1
	fig = plt.figure(i, figsize=(fig_x,fig_y))
	plt.scatter(Z_surf, int_P1, color='r')
	#plt.scatter(int_Z, int_P2, color='b')
	plt.xlabel(r'z Coordinate (\AA)')
	#plt.ylabel(r'Z (a.u.)')
	plt.savefig('{}/{}_{}_{}_{}_int_orientation.pdf'.format(groot,nimage,nslice,force,nm))
	
	print np.max(no), np.max(int_no1), np.max(int_no1) - np.max(no), (np.max(int_no1) - np.max(no)) / np.max(no) * 100
	"""	
	fig = plt.figure(11, figsize=(fig_x,fig_y))
	plt.scatter(Z, P1, color='r')
	plt.scatter(Z, P2, color='b')
	plt.xlabel(r'z Coordinate (\AA)')
	#plt.ylabel(r'Z (a.u.)')
	

	fig1, ax1 = plt.subplots(figsize=(fig_x,fig_y))
	ax1.set_xlabel(r'z Coordinate (\AA)')
	ax1.set_ylabel(r'$n$', color='r')
	ax1.axis([0,DIM[2],np.min(no),np.max(no)])
	ax1.scatter(Z, no, color='r')
	for tl in ax1.get_yticklabels():
	    tl.set_color('r')

	ax2 = ax1.twinx()
	ax2.set_ylabel(r'Density (\AA$^{-3}$)', color='b')
	ax2.axis([0,DIM[2],np.min(DEN),np.max(DEN)])
	ax2.plot(Z, DEN, color='b')
	for tl in ax2.get_yticklabels():
	    tl.set_color('b')
	#"""

	print "COMPLETE\n"
	print 'SIMULATION TIME = {} ns'.format(nimage * 0.02) 
	#plt.show()

def surface_graph(root, model, nsite, M, csize, nslice, nm, nxy, DIM):

	import positions as pos
	import intrinsic_surface as IS

	plt.rc('text', usetex=True)
	font = {'family' : 'normal','weight' : 'bold', 'size'   : 22}
	plt.rc('font', **font)
	plt.rc('xtick', labelsize=20)
	plt.rc('ytick', labelsize=20)

	with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, nm, 2000), 'r') as infile:
		_, _, int_av_mol_den, _, _, _ = np.loadtxt(infile)

	xat, yat, zat = ut.read_positions("{}/{}_{}_surface10.rst".format(root, model.lower(), csize), nsite)
	xR, yR, zR = ut.centre_mass(xat, yat, zat, nsite, M)

	with file('{}/DATA/DEN/{}_{}_{}_{}_PAR.txt'.format(root, model.lower(), csize, nslice, 2000), 'r') as infile:
		param = np.loadtxt(infile)
	with file('{}/DATA/ACOEFF/{}_{}_{}_{}_INTCOEFF.txt'.format(root, model.lower(), csize, nm, 10), 'r') as infile:
		auv1, auv2 = np.loadtxt(infile)
	with file('{}/DATA/ACOEFF/{}_{}_{}_{}_PIVOTS.txt'.format(root, model.lower(), csize, nm, 10), 'r') as infile:
		piv_n1, piv_n2 = np.loadtxt(infile)
	with file('{}/DATA/DIELEC/{}_{}_{}_{}_{}_{}_INTDEN.txt'.format(root, model.lower(), csize, nslice, 'exp', nm, 2000), 'r') as infile:
		int_exx, int_ezz, int_no, int_ne, int_ni = np.loadtxt(infile)

	print piv_n2

	x_range = np.linspace(0, DIM[0], nxy)
	y_range = np.linspace(0, DIM[1], nxy)
	X, Y = np.meshgrid(x_range,y_range)
	Z = np.linspace(- DIM[2]/2., DIM[2]/2., nslice)

	with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_CURVE.npz'.format(root, model.lower(), csize, nm, nxy, 10), 'r') as infile:
		npzfile = np.load(infile)
		XI1 = npzfile['XI1']
		XI2 = npzfile['XI2']

	plane1 = np.ones((nxy,nxy)) * 21.0
	plane2 = np.ones((nxy,nxy)) * 18.0
	plane3 = np.ones((nxy,nxy)) * 23.0
	colours1 = np.zeros((nxy,nxy)) 
	colours2 = np.zeros((nxy,nxy)) 
	colours3 = np.zeros((nxy,nxy)) 

	for n in xrange(nxy):
		x = x_range[n]
		for m in xrange(nxy):
			y = y_range[m]
			index1 = int((XI2[n][m] - plane1[n][m] + DIM[2]/2.) * nslice / DIM[2]) % nslice
			colours1[n][m] = int_no[index1]
			index2 = int((XI2[n][m] - plane2[n][m] + DIM[2]/2.) * nslice / DIM[2]) % nslice
			colours2[n][m] = int_no[index2]
			index3 = int((XI2[n][m] - plane3[n][m] + DIM[2]/2.) * nslice / DIM[2]) % nslice
			colours3[n][m] = int_no[index3]

	piv_xH1 = [xat[int(piv+1)] for piv in piv_n2]
	piv_yH1 = [yat[int(piv+1)] for piv in piv_n2]
	piv_zH1 = [zat[int(piv+1)]-zR for piv in piv_n2]
	
	piv_xH2 = [xat[int(piv+2)] for piv in piv_n2]
	piv_yH2 = [yat[int(piv+2)] for piv in piv_n2]
	piv_zH2 = [zat[int(piv+2)]-zR for piv in piv_n2]  

	piv_x = [xat[int(piv)] for piv in piv_n2]
	piv_y = [yat[int(piv)] for piv in piv_n2]
	piv_z = [zat[int(piv)]-zR for piv in piv_n2]

	max_col = np.max((colours1, colours2, colours3))
	colours1 = colours1 -1 / (max_col - 1)
	colours2 = colours2 -1 / (max_col - 1)
	colours3 = colours3 -1 / (max_col - 1)

	fig = plt.figure(1)
	N = len(piv_n2)
	ax = fig.gca(projection='3d')
	plane = ax.plot_surface(X, Y, plane1, shade=False, rstride=1, cstride=1, linewidth=0, alpha=0.6, facecolors=plt.cm.Reds(colours1))
	#p1 = ax.plot_surface(X, Y, plane2, shade=False, rstride=1, cstride=1, linewidth=0, alpha=0.6, facecolors=plt.cm.Reds(colours2))
	#p1 = ax.plot_surface(X, Y, plane3, shade=False, rstride=1, cstride=1, linewidth=0, alpha=0.8, facecolors=plt.cm.Reds(colours3))
	for i in xrange(N):
		ax.plot([piv_y[i], piv_yH1[i]], [piv_x[i], piv_xH1[i]], [piv_z[i], piv_zH1[i]], c='black')
		ax.plot([piv_y[i], piv_yH2[i]], [piv_x[i], piv_xH2[i]], [piv_z[i], piv_zH2[i]], c='black')
	ax.scatter(piv_yH1[:N], piv_xH1[:N], piv_zH1[:N], s=40, c='white', alpha=1)
	ax.scatter(piv_yH2[:N], piv_xH2[:N], piv_zH2[:N], s=40, c='white', alpha=1)
	ax.scatter(piv_y[:N], piv_x[:N], piv_z[:N], s=75, c='r', alpha=1)
	surface = ax.plot_surface(X, Y, XI2, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.4, color='blue')
	#p1 = ax.plot_wireframe(X, Y, Eta2, color='grey')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	m = cm.ScalarMappable(cmap=cm.Reds)
	m.set_array(np.linspace(1, max_col, 10))
	cbar = plt.colorbar(m, shrink=0.8)
	cbar.set_label(r'$n$')
	plt.show()

def main():

	#"""
	model = 'TIP4P2005'
	nsite, AT, Q, M, LJ = ut.get_param(model)
	lslice = 0.05 * LJ[1]
	csize = 50
	nimage = 400
	cutoff = 18
	T = 298
	folder = 'SURFACE'

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	root = '/data/fl7g13/AMBER/WATER/TIP4P2005/T_{}_K/CUT_{}_A/TIP4P2005_50/{}'.format( T, cutoff, folder)

	natom, nmol, DIM = ut.read_atom_mol_dim("{}/tip4p2005_50_surface0".format(root))
	nslice = int(DIM[2] / lslice)
	nm = int(DIM[0] / (LJ[1]))

	surface_graph(root, model, nsite, M, csize, nslice, nm, 30, DIM)

	"""
	for cutoff in [8, 18]:
		root1 = '/data/fl7g13/AMBER/WATER/TIP4P2005/T_{}_K/CUT_{}_A/TIP4P2005_50/{}'.format( T, cutoff, folder)
		natom, nmol, DIM = ut.read_atom_mol_dim("{}/tip4p2005_50_surface0".format(root1))
		nslice = int(DIM[2] / lslice)
		int_Z = np.linspace(-(DIM[2]/2.)/LJ[1], (DIM[2]/2)/LJ[1], nslice)
		nm = int(DIM[0] / (LJ[1]))
		
		with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt'.format(root1, 'tip4p2005', 50, nslice, nm, 400), 'r') as infile:
			int_av_mass_den, int_av_atom_den, int_av_mol_den, int_av_H_den, w_den_1, w_den_2 = np.loadtxt(infile)
			
			fig = plt.figure(1)	
			plt.plot(int_Z, int_av_mol_den, label=r'O $r_c$ = {} \AA'.format(cutoff))
			fig = plt.figure(2)
			plt.plot(int_Z, int_av_H_den, label=r'H $r_c$ = {} \AA'.format(cutoff))
	
			print sp.integrate.trapz(int_av_mol_den) * DIM[0] * DIM[1] * 0.05 * LJ[1], nmol

		with file('{}/DATA/DEN/{}_{}_{}_{}_PAR.txt'.format(root1, 'tip4p2005', 50, nslice, 400), 'r') as infile:
			param = np.loadtxt(infile)
		print param

	fig = plt.figure(1)	
	plt.xlabel(r'z Coordinate (\AA $\sigma^{-1}$)')
	plt.ylabel(r'$\rho(z)$ ( \AA$^{-3}$)')
	plt.axis([-5/LJ[1],10/LJ[1],0,0.1])
	plt.legend(loc=0)

	fig = plt.figure(2)	
	plt.xlabel(r'z Coordinate (\AA $\sigma^{-1}$)')
	plt.ylabel(r'$\rho(z)$ ( \AA$^{-3}$)')
	plt.axis([-5/LJ[1],10/LJ[1],0,0.15])
	plt.legend(loc=0)
	plt.show()
	"""
	"""
	model = raw_input("What model?: ")

	nsite, Q, M, LJ = ut.get_param(model)
	lslice = 0.05 * LJ[1]
	vlim = 3
	ncube = 3

	cutoff = int(raw_input("Cutoff: (A) "))

	CSIZE = []
	nimage = int(raw_input("Number of images: "))
	ndim = int(raw_input("No. of dimensions: "))

	force = raw_input("VDW Force corrections? (Y/N): ")
	if force.upper() == 'Y': folder = 'SURFACE_2'
	else: 
		folder = 'SURFACE'
		force = 'N' 
	suffix = 'surface'
	nxy = 30

	a_type = raw_input("Polarisability Parameter type? (exp, ame, abi)?: ")	

	for i in xrange(ndim):
		CSIZE.append(5 * i + 35)
		if model.upper() == 'ARGON': root = '/data/fl7g13/AMBER/{}/CUT_{}_A/{}_{}/{}'.format(model.upper(), cutoff, model.upper(), CSIZE[i], folder.upper())
		else: root = '/data/fl7g13/AMBER/WATER/{}/CUT_{}_A/{}_{}/{}'.format(model.upper(), cutoff, model.upper(), CSIZE[i], folder.upper())
		
		natom, nmol, DIM = ut.read_atom_mol_dim("{}/{}_{}_{}0".format(root, model.lower(), CSIZE[i], suffix))

		nslice = int(DIM[2] / lslice)
		nm = int(DIM[0] / (LJ[1]))
		
		groot = "/home/fl7g13/Documents/Figures/{}_{}_{}".format(model.upper(),CSIZE[i],cutoff)
		if not os.path.exists(groot): os.mkdir(groot)

		print_graphs(root, groot, model, CSIZE[i], nslice, nimage, a_type, force, nm, DIM)
"""
"""
ngrid = 30
x_range = np.linspace(0, DIM[0], ngrid)
y_range = np.linspace(0, DIM[1], ngrid)
X, Y = np.meshgrid(x_range,y_range)

Eta1 = np.zeros((ngrid,ngrid))
Eta2 = np.zeros((ngrid,ngrid))
Eta3 = np.zeros((ngrid,ngrid))

surface1 = np.ones((ngrid,ngrid)) * SURF[0]
surface2 = np.ones((ngrid,ngrid)) * SURF[1]

for n in xrange(ngrid):
	x = x_range[n]
	for m in xrange(ngrid):
		y = y_range[m]
		Eta1[n][m] = xi(x, y, qu, auv1, DIM)
		Eta2[n][m] = xi(x, y, qu, auv2, DIM)
		Eta3[n][m] = xi(x, y, qu, auv3, DIM)

diff = map (lambda x: Eta1[int(x/ngrid)][x%ngrid] + Eta2[int(x/ngrid)][x%ngrid], xrange(ngrid**2))
print np.max(Eta1), np.min(Eta1), np.max(Eta2), np.min(Eta2)
print np.mean(Eta1) - SURF[0], np.mean(Eta2) - SURF[1]

X1 = [xat[i][piv_n1[ns]] for ns in xrange(len(piv_n1))]
Y1 = [yat[i][piv_n1[ns]] for ns in xrange(len(piv_n1))]
Z1 = [zat[i][piv_n1[ns]]-zcom for ns in xrange(len(piv_n1))]

X2 = [xat[i][piv_n2[ns]] for ns in xrange(len(piv_n2))]
Y2 = [yat[i][piv_n2[ns]] for ns in xrange(len(piv_n2))]
Z2 = [zat[i][piv_n2[ns]]-zcom for ns in xrange(len(piv_n2))]

X3 = [xat[i][piv_ncom[ns]] for ns in xrange(len(piv_ncom))]
Y3 = [yat[i][piv_ncom[ns]] for ns in xrange(len(piv_ncom))]
Z3 = [zat[i][piv_ncom[ns]]-zcom for ns in xrange(len(piv_ncom))]

fig = plt.figure(0)
plt.scatter(xrange(ngrid**2),diff)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
#p1 = ax.plot_surface(X, Y, Eta, rstride=1, cstride=1, linewidth=0, antialiased=False)
p1 = ax.plot_wireframe(X, Y, Eta1)
p1 = ax.plot_wireframe(X, Y, Eta2)
p1 = ax.plot_wireframe(X, Y, Eta3)
#p1 = ax.plot_wireframe(X, Y, surface1,color='g')
#p1 = ax.plot_wireframe(X, Y, surface2,color='g')
#p2 = ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, linewidth=0, antialiased=False, color='r')
p4 = ax.scatter(Y1, X1, Z1, color='red')
p4 = ax.scatter(Y2, X2, Z2, color='black')
p4 = ax.scatter(Y3, X3, Z3, color='green')

#plt.title("phi")
#plt.legend()
plt.show()
#"""
