"""
*************** DIELECTRIC PROFILE MODULE *******************

Calculates electrostatic properties such as dielectric and
refractive index profiles.

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


def theta_integrate_y(K, x, DIM, auv, nm, zcom):

	a, b = spin.quad(lambda y: np.sin(np.arccos(np.dot(K, normal_vector(x,y,nm,auv,DIM,zcom))))**2, 0, DIM[1])
	return a

def theta_integrate_x_y(K, DIM, auv, nm, zcom):

	a, b = spin.quad(lambda x: integrate_y(K, x, DIM, auv, nm, zcom), 0, DIM[0])
	return a 

def den_integrate_y(z, x, DIM, auv, nm, zcom):

	a, b = spin.quad(lambda y: int_av_mol_den(z - xi(x, y, nm, zcom)) , 0, DIM[1])
	return a

def den_integrate_x_y(K, DIM, auv, nm, zcom):

	a, b = spin.quad(lambda x: integrate_y(K, x, DIM, auv, nm, zcom), 0, DIM[0])
	return a 


def normal_vector(z, dx, dy, DIM, zcom):

	T = ut.local_frame_surface(dx, dy, z, zcom)
	T = ut.unit_vector(np.sum(T[0]), np.sum(T[1]), np.sum(T[2]))
	
	return T

def surface_ne(root, model, csize, nm, nxy, int_exx, int_ezz, DIM, nimage, thetai):

	nslice = len(int_exx)
	X = np.linspace(0, DIM[0], nxy)
	Y = np.linspace(0, DIM[1], nxy)
	Z = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

	angle1 = np.zeros((nimage,nxy,nxy))
	angle2 = np.zeros((nimage,nxy,nxy))

	K = ut.unit_vector(0 , np.cos(thetai), np.sin(thetai))
	unit = np.sqrt(1/2.)
	print K, np.arccos(np.dot(K, ([0,1,0]))) * 180 / np.pi, thetai * 180 / np.pi

	T_int_anis = np.zeros(nslice)

	print 'PROCESSING SURFACE CURVATURE'
	for i in xrange(nimage):
		sys.stdout.write("CALCULATING {} out of {} ANGLES\r".format(i+1, nimage) )
		sys.stdout.flush()
		
		with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_{}_CURVE.npz'.format(root, model.lower(), csize, nslice, nm, nxy, i), 'r') as infile:
			npzfile = np.load(infile)
			XI1 = npzfile['XI1']
			XI2 = npzfile['XI2']
			DX1 = npzfile['DX1']
			DY1 = npzfile['DY1']
			DX2 = npzfile['DX2']
			DY2 = npzfile['DY2']
		for j in xrange(nxy):
			x = X[j]
			for k in xrange(nxy):
				y = Y[k]
				angle1[i][j][k] = np.arccos(np.dot(K, normal_vector(Eta1[j][k], Dx1[j][k], Dy1[j][k], DIM,)))
				angle2[i][j][k] = np.arccos(np.dot(K, normal_vector(Eta2[j][k], Dx2[j][k], Dy2[j][k], DIM)))

	print "\n"

	for i in xrange(nimage):
		sys.stdout.write("READJUSTING {} out of {} ORIENTATIONS\r".format(i+1, nimage) )
		sys.stdout.flush()
		"""
		if os.path.exists('{}/DATA/DIELEC/{}_{}_{}_{}_{}_{}_ANIS.txt'.format(root, model.lower(), csize, nslice, nm, nxy, i)): A = 1
			with file('{}/DATA/DIELEC/{}_{}_{}_{}_{}_{}_ANIS.txt'.format(root, model.lower(), csize, nslice, nm, nxy, i)) as infile:
				T_int_anis += np.loadtxt(infile) / nimage

		else:
		"""

		plane1 = 0
		plane2 = 0

		for j in xrange(nxy):
			x = X[j]
			for k in xrange(nxy):
				y = Y[k]
				plane1 += np.sin(angle1[i][j][k])**2  / (nxy**2)
				plane2 += np.sin(angle2[i][j][k])**2  / (nxy**2)

		print plane1 * 180 / np.pi, plane2 * 180 / np.pi

		int_anis = np.zeros(nslice)

		for n in xrange(nslice):
			prefac = (int_ezz[n] - int_exx[n]) / int_ezz[n]
			if Z[n] < 0: int_anis[n] += (1 - prefac * plane1) 
			else: int_anis[n] += (1 - prefac * plane2)

		int_anis2 = np.zeros(nslice)

		for n in xrange(nslice):
			prefac = (int_ezz[n] - int_exx[n]) / int_ezz[n]
			for j in xrange(nxy):
				x = X[j]
				for k in xrange(nxy):
					y = Y[k]
					if Z[n] < 0: int_anis2[n] += (1 - prefac * np.sin(angle1[i][j][k])**2) / (nxy**2)
					else: int_anis2[n] += (1 - prefac * np.sin(angle2[i][j][k])**2) / (nxy**2)

		print np.sum(int_anis - int_anis2)
		T_int_anis += int_anis / nimage
		with file('{}/DATA/DIELEC/{}_{}_{}_{}_{}_{}_ANIS.txt'.format(root, model.lower(), csize, nslice, nm, nxy, i), 'w') as outfile:
			np.savetxt(outfile, (int_anis), fmt='%-12.6f')
	print "\n"

	return T_int_anis


def dielectric_refractive_index(directory, model, csize, AT, sigma, mol_sigma, nslice, nframe, a_type, DIM):

	atom_types = list(set(AT))
	n_atom_types = len(atom_types)

	Z = np.linspace(0, DIM[2], nslice)
        Z2 = np.linspace(-DIM[2]/2., DIM[2]/2., nslice)

        lslice = DIM[2] / nslice
        ur = 1 #- 9E-6
        angle = 52.9*np.pi/180.
	a = ut.get_polar_constants(model, a_type)

	with file('{}/DATA/DEN/{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
                av_density = np.loadtxt(infile)

	mol_den = av_density[-1]

	if model.upper() == 'ARGON':
		axx = np.ones(nslice) * a
		azz = np.ones(nslice) * a
	else:
		with file('{}/DATA/EULER/{}_{}_{}_{}_EUL.txt'.format(directory, model.lower(), a_type, nslice, nframe), 'r') as infile:
			axx, azz, _, _, _, _, _ = np.loadtxt(infile)

	exx = np.array([(1 + 8 * np.pi / 3. * mol_den[n] * axx[n]) / (1 - 4 * np.pi / 3. * mol_den[n] * axx[n]) for n in range(nslice)])
        ezz = np.array([(1 + 8 * np.pi / 3. * mol_den[n] * azz[n]) / (1 - 4 * np.pi / 3. * mol_den[n] * azz[n]) for n in range(nslice)])

        no = np.sqrt(ur * exx)
        ni = np.sqrt(ur * ezz)

	popt, pcov = curve_fit(ut.den_func, Z, no, [1., 1., DIM[2]/2., DIM[2]/4., 2.])
        param = np.absolute(popt)
        no_sm = map (lambda x: ut.den_func(x, param[0], 1, param[2], param[3], param[4]), Z)
	popt, pcov = curve_fit(ut.den_func, Z, ni, [1., 1., DIM[2]/2., DIM[2]/4., 2.])
        param = np.absolute(popt)
	ni_sm = map (lambda x: ut.den_func(x, param[0], 1, param[2], param[3], param[4]), Z)

	with file('{}/DATA/DIELEC/{}_{}_{}_{}_DIE.txt'.format(directory, model.lower(), a_type, nslice, nframe), 'w') as outfile:
		np.savetxt(outfile, (exx, ezz), fmt='%-12.6f')
	with file('{}/DATA/DIELEC/{}_{}_{}_{}_ELLIP_NO.txt'.format(directory, model.lower(), a_type, nslice, nframe), 'w') as outfile:
		np.savetxt(outfile, (no_sm, ni_sm), fmt='%-12.6f')

	print "{} {} {} COMPLETE\n".format(directory, model.upper(), csize)


