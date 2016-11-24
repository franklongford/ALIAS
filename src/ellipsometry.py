"""
*************** ELLIPSOMETRY MODULE *******************

PROGRAM INPUT:
	       

PROGRAM OUTPUT: 
		
		

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

sys.path.append('/data/fl7g13/tmm-0.1.5/')
import tmm_core
import csv


def n_func(x, a, b, c): return a*(x-b)**2 + c

def den_func(z, nv, nl, z0, d): return 0.5 * (nl + nv) - 0.5 * (nl - nv) * np.tanh((z - z0)/ (2*d))

def den_func2(z, m, s1, s2, d): return 0.5 * (m + 1) - 0.5 * (m - 1) * (np.tanh((z - s1)/ (2*d)) * np.tanh((z - s2)/ (2*d)))

def increase_deriv(m, n, Z):

	range_ = Z[-1] - Z[0]
	nslice = len(Z)

	new_n = copy.copy(n)

	for i in xrange(1,nslice):
		dn = n[i]-n[i-1]
		new_n[i] = n[i-1] + dn * m

	return new_n 

def main(TYPE):

	"INPUT DATA"

	if TYPE.upper() == 'TEST':

		#model = 'SPCE'
		model = 'METHANOL'
		model = 'TIP4P2005'
		nsite, AT, Q, M, LJ = ut.get_param(model)
		sigma = np.max(LJ[1])
		lslice = 0.05 * sigma
		print lslice / 10.
		cutoff = 18
		vlim = 3
		ncube = 3
		CSIZE = ([50])
		#CSIZE = ([40])
		ndim = len(CSIZE)
		nimage = 2000
		a_type = 'exp'
		suffix = "surface"
		nfolder = 1
		T = 298
		force = 0
		csize = CSIZE[0]
		folder = 'surface'
		if model == 'TIP4P2005':root = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A/{}_{}/{}'.format(model.upper(), T, cutoff, model.upper(), csize, folder.upper())
		else: root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/{}_{}/{}'.format(model.upper(), T, cutoff, model.upper(), csize, folder.upper())

	degree = np.pi/180.
	start_lam =200
	end_lam = 1000
	lambda_list = np.linspace(start_lam,end_lam,100)
	COLOUR = ['b', 'g', 'r', 'c', 'm', 'saddlebrown', 'navy']
	nruns = 9

	theta1 = np.arange(40,71)
	theta2 = np.arange(50, 55.01, 0.1)
	theta3 = np.arange(52, 54.01, 0.1)
	standard_theta = np.arange(52, 54.08, 0.1)

	psi = np.zeros((nruns, len(theta3)))
	delta = np.zeros((nruns, len(theta3)))
	av_psi = np.zeros(len(theta3)+7)
	av_delta = np.zeros(len(theta3)+7)

	theta = np.array((theta3+0.2, theta3, theta3, theta3+0.2, theta3, theta3, theta3+0.7, theta3+0.7, theta3+0.2))

	with file("ellipsometry_data_psi.csv", 'r') as infile:
		psi[0], psi[1], psi[2], _, psi[3], psi[4], psi[5], psi[6], psi[7], psi[8] = np.loadtxt(infile)
	with file("ellipsometry_data_delta.csv", 'r') as infile:
		delta[0], delta[1], delta[2], _, delta[3], delta[4], delta[5], delta[6], delta[7], delta[8] = np.loadtxt(infile)

	for i in xrange(9):
		for j in xrange(len(theta3)):
			the = theta[j][i]
			index = (the - 52) / 0.1
			print the, index, psi[i][j]
			av_psi[index] += psi[j][i] / 9.
			av_delta[index] += delta[j][i] / 9.
	
	print standard_theta, av_psi, av_delta
	sys.exit()

	n = [1.396+1.10E-7j, 1.373+4.9E-8j, 1.362+3.35E-8j, 1.354+2.35E-8j, 1.349+1.6E-8j, 1.346+1.08E-8j, 1.343+6.5E-9j, 1.341+3.5E-9j, 
	     1.339+1.86E-9j, 1.338+1.3E-9j, 1.337+1.02E-9j, 1.336+9.35E-10j, 1.335+1.00E-9j, 1.334+1.32E-9j, 1.333+1.96E-9j, 
             1.333+3.60E-9j, 1.332+1.09E-8j, 1.332+1.39E-8j, 1.331+1.64E-8j, 1.331+2.23E-8j, 1.331+3.35E-8j, 1.330+9.15E-8j,
	     1.330+1.56E-7j, 1.330+1.48E-7j, 1.329+1.25E-7j, 1.329+1.82E-7j, 1.329+2.93E-7j, 1.328+3.91E-7j, 1.328+4.86E-7j,
	     1.328+1.06E-6j, 1.327+2.93E-6j, 1.327+3.48E-6j, 1.327+2.89E-6j]
	k = np.linspace(start_lam,end_lam,len(n))

	print n, k

	real_param, _ = curve_fit(n_func, k , np.real(n), [1.331, 700, 1.331])
	real_n = map (lambda x: real_param[0]*(x-real_param[1])**2 + real_param[2], lambda_list)
	imag_param, _ = curve_fit(n_func, k , np.imag(n), [5E-13, 450, 5E-13])
	imag_n = map (lambda x: imag_param[0]*(x-imag_param[1])**2 + imag_param[2], lambda_list)

	"""
	plt.figure(0)
	plt.plot(k,np.imag(n))
	plt.plot(lambda_list, imag_n)
	#plt.axis([300, 700, imag_param[2], 5E-8])
	
	plt.figure(1)
	plt.plot(k,np.real(n))
	plt.plot(lambda_list, real_n)
	#plt.axis([start_lam, end_lam, 1.330, 1.400])
	#plt.show()
	"""

	natom, nmol, DIM = ut.read_atom_mol_dim("{}/{}_{}_{}0".format(root, model.lower(), csize, suffix))
	nslice = int(DIM[2] / lslice)
	nm = int(DIM[0] / sigma)
	Z = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)
	nxy = 30

	with file('{}/DATA/DEN/{}_{}_{}_{}_PAR.txt'.format(root, model.lower(), csize, nslice, nimage), 'r') as infile:
		param = np.loadtxt(infile)

	with file('{}/DATA/DIELEC/{}_{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, a_type, nimage), 'r') as infile:
		exx, ezz, no, ne, ni = np.loadtxt(infile)

	with file('{}/DATA/DIELEC/{}_{}_{}_{}_{}_{}_INTDEN.txt'.format(root, model.lower(), csize, nslice, a_type, nm, nimage), 'r') as infile:
		int_exx,int_ezz,int_no,int_ne,int_ni = np.loadtxt(infile)
	
	with file('{}/DATA/DIELEC/{}_{}_{}_{}_{}_{}_WINTDEN.txt'.format(root, model.lower(), csize, nslice, a_type, nm, nimage), 'r') as infile:
		_, _, int_no1, int_ne1, int_ni1, _, _, _, _, _ = np.loadtxt(infile)
	
	if model.upper() == 'METHANOL':
		start_z = 10
		end_z = 50
	else:
		start_z = 15
		end_z = 40	

	#no = map (lambda x: no[x] + 0j, range(len(no))[int(nslice/6.):int(nslice*1/2.)])
	no = no[int(nslice * start_z / DIM[2]):int(nslice * end_z / DIM[2])]
	int_no = int_no[int(nslice * (DIM[2]/2+(start_z-24.5))/DIM[2]):int(nslice *(DIM[2]/2+(end_z-24.5))/DIM[2])]
	print int_no

	int_no1 = int_no1[int(nslice * start_z / DIM[2]):int(nslice * end_z / DIM[2])]
	Z1 = Z[int(nslice * start_z / DIM[2]):int(nslice * end_z / DIM[2])]
	Z2 = Z[int(nslice * (DIM[2]/2-7.5)/DIM[2]):int(nslice *(DIM[2]/2+6.5)/DIM[2])]

	print no

	param, _ = curve_fit(den_func, Z1 , no, [1, no[-1], -20, 1 ])
	no = map (lambda x: den_func(x, param[0], param[1], param[2], param[3]), Z1)

	print no
	function1 = map(lambda x: den_func2(Z1[x], 1.15, param[2]-1, param[2]+1, 0.5), range(len(Z1)))
	function2 = map(lambda x: den_func2(Z1[x], 1.075, -18, -17, 0.5), range(len(Z1)))
	#int_no1_test = map(lambda x: function1[x] * function2[x] * int_no1[x], range(len(Z1)))
	int_no1_test = increase_deriv(6.5, int_no1, Z1)

	"""
	plt.plot(Z1, int_no)
	plt.plot(Z1, int_no1)
	#plt.plot(Z1, function1)
	#plt.plot(Z1, function2)
	plt.plot(Z1, no)
	plt.show()
	"""

	start_d = 59
	end_d = 60
	n_d = 20

	start_n = 0.980
	end_n = 1.05
	n_n = 10

	d_0 = 0.10 * sigma
	n_0 = 1.00

	PSI1 = []
	DELTA1 = []
	PSI2 = []
	DELTA2 = []
	int_PSI1 = []
	int_DELTA1 = []
	int_PSI2 = []
	int_DELTA2 = []
	int_PSI3 = []
	int_DELTA3 = []
	int_PSI4 = []
	int_DELTA4 = []
	int_PSI5 = []
	int_DELTA5 = []

	percent1 = 5
	percent2 = 5

	scan_type = 0

	fig_x = 22
	fig_y = 14

	plt.rc('text', usetex=True)
	font = {'family' : 'normal','weight' : 'bold', 'size'   : 22}
	plt.rc('font', **font)
	plt.rc('xtick', labelsize=20)
	plt.rc('ytick', labelsize=20)
	
	groot = "/home/fl7g13/Documents/Figures/{}_{}_{}".format(model.upper(),CSIZE[0],cutoff)
	if not os.path.exists(groot): os.mkdir(groot)

	#for d_0 in np.linspace(start_d, end_d, n_d):
	if scan_type == 0: 
		lam = 514.5
		n = 1.3344
		n = 1.3295
	
		Bangle = np.real(np.arctan(n)) / degree

		start_angle = 52 #np.real(Bangle) - 0.5 #
		end_angle = 54 #np.real(Bangle) + 0.5 #
		n_angle = 200

		ANGLE = np.linspace(start_angle * degree, end_angle * degree, n_angle)

		no1 = [1, n]
		intno1 = [1, n * (1+percent1/100.), n]
		intno2 = [1, n * (1-percent2/100.), n]
		#intno1 = [1, n * (1+percent/20.), 1.02, n * (1+percent/40.), n]
		D = [np.inf, np.inf]
		intD1 = [np.inf, d_0, np.inf]
		intD2 = [np.inf, d_0, np.inf]
		#intD2 = [np.inf, d_0, 0.5*d_0, 0.75*d_0, np.inf]
		print param[2]
		surf = param[2]

		intZ1 = [Z1[0], surf, surf, surf+d_0*10, surf+d_0*10, Z1[-1]]
		intn1 = [1, 1, intno1[1], intno1[1],  intno1[2], intno1[2]]
		intZ2 = [Z1[0], surf, surf, surf+d_0*10, surf+d_0*10, Z1[-1]]
		intn2 = [1, 1, intno2[1], intno2[1],  intno2[2], intno2[2]]

		#intZ1 = [Z1[0], surf, surf, surf+intD2[1]*10, surf+intD2[1]*10, surf+(intD2[1]+intD2[2])*10, surf+(intD2[1]+intD2[2])*10, surf+(intD2[1]+intD2[2]+intD2[3])*10, surf+(intD2[1]+intD2[2]+intD2[3])*10, Z1[-1]]
		#intn1 = [1, 1, intno1[1], intno1[1], intno1[2], intno1[2], intno1[3], intno1[3], intno1[4], intno1[4]]
		
		int_no[-1] = np.max([param[0], param[1]])
		no2 = no #+ 1j * (no - 1) / (no[-1] - 1) * np.imag(n[13])
		intno3 = int_no #+ 1j * (int_no - 1) / (int_no[-1] - 1) * np.imag(n[13])
		intno4 = int_no1
	
		#no[0] = 1
		#intno[0] = 1
		#intno1[0] = 1

		print param

		#no2[-1] = np.max([param[0], param[1]])
		#intno3[-1] = np.max([param[0], param[1]])

		plt.figure(10, figsize=(fig_x,fig_y))
		plt.plot([Z1[0], surf, surf, Z1[-1]], [1, 1, no1[1], no1[1]], label='Ideal', linestyle='dashdot')
		plt.plot(intZ1, intn1, label='Higher', linestyle='dashed')
		plt.plot(intZ2, intn2, label='Lower', linestyle='dashed')
		
		#plt.legend(loc=1)
		plt.xlabel(r'z Coordinate (\AA)')
		plt.ylabel(r'n (a.u.)')
		#plt.figure(11, figsize=(fig_x,fig_y))
		plt.plot(Z1, no,  label=r'$\rho(z)$')
		#plt.plot(np.linspace(Z[0], Z[-1], len(int_no1)), int_no1,  label=r'$\tilde{\rho}(z)$')
		plt.plot(Z1, int_no,  label=r'$\tilde{\rho}(z)$')
		plt.plot(Z1, int_no1,  label=r'$\tilde{\rho}_e(z)$')
		#plt.plot(Z1, int_no1_test,  label=r'$m\tilde{\rho}_e(z)$')
		plt.legend(loc=2)
		plt.axis([-DIM[2]/2+start_z, -DIM[2]/2+end_z, 1, 2.2])
		plt.savefig('{}/{}_{}_{}_{}_n_profile.pdf'.format(groot,nimage,nslice,force,nm))
		

		D2 = np.ones(len(no)) * lslice / 10. 
		D2[0] = np.inf
		D2[-1] = np.inf

		D3 = np.ones(len(int_no1)) * lslice / 10. 
		D3[0] = np.inf
		D3[-1] = np.inf

		for angle in ANGLE:
			sys.stdout.write("PROCESSING ANGLE {} out of {}\r".format(angle, ANGLE[-1]))
			sys.stdout.flush()
			reg_e1 = tmm_core.ellips(no1, D, angle, lam)
			int_e1 = tmm_core.ellips(intno1, intD2, angle, lam)
			int_e2 = tmm_core.ellips(intno2, intD1, angle, lam)

			reg_e2 = tmm_core.ellips(no2, D2, angle, lam)
			int_e3 = tmm_core.ellips(intno3, D2, angle, lam)
			int_e4 = tmm_core.ellips(intno4, D2, angle, lam)
			int_e5 = tmm_core.ellips(int_no1_test, D2, angle, lam)

			"""
			PSI.append(reg_e['psi'] / degree)
			DELTA.append(-reg_e['Delta']/ degree)
			int_PSI1.append(int_e1['psi'] /degree)
			int_DELTA1.append(-int_e1['Delta']  / degree)
			int_PSI2.append(int_e2['psi']  /degree)
			int_DELTA2.append(-int_e2['Delta'] / degree)
			"""
			PSI1.append((reg_e1['psi']) / degree)
			DELTA1.append((-reg_e1['Delta'] + np.pi) / degree)
			PSI2.append((reg_e2['psi']) / degree)
			DELTA2.append((-reg_e2['Delta'] + np.pi) / degree)
			int_PSI1.append((int_e1['psi']) /degree)
			int_DELTA1.append((-int_e1['Delta'] + np.pi) / degree)
			int_PSI2.append((int_e2['psi']) /degree)
			int_DELTA2.append((-int_e2['Delta'] + np.pi) / degree)
			int_PSI3.append((int_e3['psi']) /degree)
			int_DELTA3.append((-int_e3['Delta'] + np.pi) / degree)
			int_PSI4.append((int_e4['psi']) /degree)
			int_DELTA4.append((-int_e4['Delta'] + np.pi) / degree)
			int_PSI5.append((int_e5['psi']) /degree)
			int_DELTA5.append((-int_e5['Delta'] + np.pi) / degree)
			#"""

	elif scan_type == 1: 
		angle = 52.9 * degree
		for i in xrange(len(k)):
			lam = -k[i]

			no1 = [1, n[i]]
			intno1 = [1, n[i] * (1+percent*10/100.), n[i]]
			intno2 = [1, n[i] * (1-percent/100.), n[i]]
			D = [np.inf, np.inf]
			intD1 = [np.inf, d_0, np.inf]

			no[0] = 1.0
			intno1[0] = 1.0

			D2 = np.ones(len(no)) * lslice / 10. 
			D2[0] = np.inf
			D2[-1] = np.inf

			D3 = np.ones(len(int_no1)) * lslice / 10. 
			D3[0] = np.inf
			D3[-1] = np.inf

			no2 = (no - 1) / (no[-1] - 1) * (n[i] - 1) + 1
			intno3 = (int_no1 - 1) / (int_no1[-1] - 1) * (n[i] - 1) + 1
			intno4 = (int_no2 - 1) / (int_no2[-1] - 1) * (n[i] - 1) + 1

			no2[0] = 1.0
			intno3[0] = 1.0
			intno4[0] = 1.0

			print n[i], no2[-1], intno3[-1], intno4[-1]

			if i == 0:
				surf = param[2]-DIM[2]/2 + 0.5

				intZ1 = [Z[0], surf, surf, surf+d_0*10, surf+d_0*10, Z[-1]]
				intn1 = [1, 1, intno1[1], intno1[1],  intno1[2], intno1[2]]
				intZ2 = [Z[0], surf, surf, surf+d_0*10, surf+d_0*10, Z[-1]]
				intn2 = [1, 1, intno2[1], intno2[1],  intno2[2], intno2[2]]
		
				plt.figure(10)
				plt.plot([Z[0], surf, surf, Z[-1]], [1, 1, no1[1], no1[1]], label='Ideal {} nm'.format(lam))
				plt.plot(intZ1, intn1, label='Higher {} nm'.format(lam))
				plt.plot(intZ2, intn2, label='Lower {} nm'.format(lam))
				plt.plot(Z, no2, label='Slab {} nm'.format(lam))
				plt.plot(np.linspace(Z[0], Z[-1], len(int_no1)), intno3, label='Intrinsic {} nm'.format(lam))
				plt.plot(Z, intno4, label='Smoothed Intrinsic {} nm'.format(lam))
				plt.legend(loc=2)

			reg_e1 = tmm_core.ellips(no1, D, angle, lam)
			int_e1 = tmm_core.ellips(intno1, intD1, angle, lam)
			int_e2 = tmm_core.ellips(intno2, intD1, angle, lam)

			reg_e2 = tmm_core.ellips(no2, D2, angle, lam)
			int_e3 = tmm_core.ellips(intno3, D3, angle, lam)
			int_e4 = tmm_core.ellips(intno4, D2, angle, lam)
			#"""
		
			"""
			PSI.append(reg_e['psi'] / degree)
			DELTA.append(-reg_e['Delta']/ degree)
			int_PSI1.append(int_e1['psi'] /degree)
			int_DELTA1.append(-int_e1['Delta']  / degree)
			int_PSI2.append(int_e2['psi']  /degree)
			int_DELTA2.append(-int_e2['Delta'] / degree)
			"""
			PSI1.append((reg_e1['psi']) / degree)
			DELTA1.append((-reg_e1['Delta'] + np.pi) / degree)
			PSI2.append((reg_e2['psi']) / degree)
			DELTA2.append((-reg_e2['Delta'] + np.pi) / degree)
			int_PSI1.append((int_e1['psi']) /degree)
			int_DELTA1.append((-int_e1['Delta'] + np.pi) / degree)
			int_PSI2.append((int_e2['psi']) /degree)
			int_DELTA2.append((-int_e2['Delta'] + np.pi) / degree)
			int_PSI3.append((int_e3['psi']) /degree)
			int_DELTA3.append((-int_e3['Delta'] + np.pi) / degree)
			int_PSI4.append((int_e4['psi']) /degree)
			int_DELTA4.append((-int_e4['Delta'] + np.pi) / degree)
			#"""


	plt.figure(5, figsize=(fig_x,fig_y))
	if scan_type == 0:
		plt.plot(ANGLE / degree,PSI1, label='Ideal', linestyle='dashdot')
		plt.plot(ANGLE / degree,int_PSI1, label='Higher', linestyle='dashed')
		plt.plot(ANGLE / degree,int_PSI2, label='Lower', linestyle='dashed')
		plt.plot(ANGLE / degree,PSI2, label=r'$\rho(z)$')
		plt.plot(ANGLE / degree,int_PSI3, label=r'$\tilde{\rho}(z)$')
		plt.plot(ANGLE / degree,int_PSI4, label=r'$\tilde{\rho}_e(z)$')
		#plt.plot(ANGLE / degree,int_PSI5, label=r'$m\tilde{\rho}_e(z)$')
		#plt.scatter(theta, psi1, label='Experimental', c='b', linestyle='dashed', )
		for i in xrange(nruns):
			plt.plot(theta[i], psi[i], label='Experimental {}'.format(i), linestyle='dotted')
		plt.plot(standard_theta, av_psi, label='Experimental Average'.format(i), linestyle='solid')
		plt.xlabel(r'$\theta_i$ ($^\circ$)')
		plt.axis([start_angle, end_angle, 0, 2])
	if scan_type == 1:
		plt.plot(k,PSI1, label='Ideal {} nm'.format(514.5))
		plt.plot(k,int_PSI1, label='Higher {} nm'.format(514.5))
		plt.plot(k,int_PSI2, label='Lower {} nm'.format(514.5))
		plt.plot(k, PSI2, label='Slab 514.5 nm')
		plt.plot(k, int_PSI3, label='Intrinsic 514.5 nm')
		plt.plot(k, int_PSI4, label='Smoothed Intrinsic 514.5 nm')
		plt.xlabel('k (nm)')		
	#plt.xlabel('angle ($^\circ$)')
	#plt.ylabel('$\Psi$')
	#plt.title('Amplitude $\Psi$')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_{}_amplitude.pdf'.format(groot,nimage,nslice,force,nm))
	

	plt.figure(6, figsize=(fig_x,fig_y))
	if scan_type == 0:
		plt.plot(ANGLE / degree,DELTA1, label='Ideal', linestyle='dashdot')
		plt.plot(ANGLE / degree, int_DELTA1, label='Higher', linestyle='dashed')
		plt.plot(ANGLE / degree, int_DELTA2, label='Lower', linestyle='dashed')
		plt.plot(ANGLE / degree, DELTA2, label=r'$\rho(z)$')
		plt.plot(ANGLE / degree, int_DELTA3, label=r'$\tilde{\rho}(z)$')
		plt.plot(ANGLE / degree, int_DELTA4, label=r'$\tilde{\rho}_e(z)$')
		#plt.plot(ANGLE / degree, int_DELTA5, label=r'$m\tilde{\rho}_e(z)$')
		#plt.scatter(theta, delta1, label= 'Experimental 1', c='b', linestyle='dashed')
		for i in xrange(nruns):
			plt.plot(theta[i], delta[i], label='Experimental {}'.format(i), linestyle='dotted')
		plt.plot(standard_theta, av_delta, label='Experimental Average'.format(i), linestyle='solid')
		plt.xlabel(r'$\theta_i$ ($^\circ$)')
		plt.xlabel(r'$\theta_i$ ($^\circ$)')
		plt.axis([start_angle, end_angle, 0, 360])
	if scan_type == 1:
		plt.plot(k,PSI1, label='Ideal'.format(514.5))
		plt.plot(k,int_PSI1, label='Higher {} nm'.format(514.5))
		plt.plot(k,int_PSI2, label='Lower {} nm'.format(514.5))
		plt.plot(k,PSI2, label='Slab 514.5 nm')
		plt.plot(k,int_PSI3, label='Intrinsic 514.5 nm')
		plt.plot(k,int_PSI4, label='Smoothed Intrinsic 514.5 nm')
		plt.xlabel('k (nm)')
		plt.axis([300, 700, 0, 360])
	#plt.ylabel('$\Delta$')
	#plt.title('Phase Shift $\Delta$ at {:3.4}$^\circ$ incidence'.format(angle/degree))
	#plt.title('Phase Shift $\Delta$ ')
	plt.legend(loc=3)
	plt.savefig('{}/{}_{}_{}_{}_delta.pdf'.format(groot,nimage,nslice,force,nm))
	plt.show()

