"""
*************** ELECTRIC FIELD MODULE *******************

Calculates electrostatic properties such as dielectric and
refractive index profiles.

***************************************************************
Created 30/4/2017 by Frank Longford

Contributors: Frank Longford

Last modified 30/4/2017 by Frank Longford
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

def charge_density(root, model, AT, Q, csize, nslice, nimage, a_type, force, nm, nxy, DIM, ow_A):

	print "PROCESSING DIELECTRIC AND REFRACTIVE INDEX PROFILES\n"

	with file('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, nimage), 'r') as infile:
		av_mass_den, av_atom_den, av_mol_den, av_H_den = np.loadtxt(infile)

	with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, nm, nimage), 'r') as infile:
		int_av_mass_den, int_av_atom_den, int_av_mol_den, int_av_H_den, w_den_1, w_den_2 = np.loadtxt(infile)

	with file('{}/DATA/DEN/{}_{}_{}_COM.txt'.format(root, model.lower(), csize, nimage), 'r') as infile:
		xR, yR, zR = np.loadtxt(infile)
		if nimage ==1: zR = [zR]
	
	with file('{}/DATA/DEN/{}_{}_{}_{}_PAR.txt'.format(root, model.lower(), csize, nslice, nimage), 'r') as infile:
		param = np.loadtxt(infile)

	Z = np.linspace(0, DIM[2], nslice)
	int_Z = np.linspace(-DIM[2]/2., DIM[2]/2., nslice)

	dz = DIM[2] / nslice
	c_den = [int_av_mol_den[n] * Q[AT.index('lp')] + int_av_H_den * Q[AT.index('H')] for n in range(nslice)]

	plt.plot(int_Z, c_den)
	plt.show()
	

	print "WRITING TO FILE..."

	with file('{}/DATA/ELEC_FIELD/{}_{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), csize, nslice, a_type, nimage), 'w') as outfile:
		np.savetxt(outfile, (c_den), fmt='%-12.6f')

	print "{} {} {} COMPLETE\n".format(root, model.upper(), csize)

def main(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, TYPE, folder, nfolder, suffix, ntraj):


	for n in range(sfolder, nfolder):
		root = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A/{}_{}/{}'.format(model.upper(), T, cutoff, model.upper(), csize, folder.upper())
		
		natom, nmol, DIM = ut.read_atom_mol_dim("{}/{}_{}_{}".format(root, model.lower(), csize, suffix))

		nslice = int(DIM[2] / lslice)
		nm = int(DIM[0] / (LJ[1]))

		if not os.path.exists("{}/DATA/ELEC_FIELD".format(root)): os.mkdir("{}/DATA/ELEC_FIELD".format(root))
		if os.path.exists('{}/DATA/ELEC_FIELD/{}_{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), CSIZE[i], nslice, a_type, nimage)):
			print '\nFILE FOUND {}/DATA/ELEC_FIELD/{}_{}_{}_{}_{}_DEN.txt'.format(root, model.lower(), CSIZE[i], nslice, a_type, nimage)
			overwrite = raw_input("OVERWRITE? (Y/N): ")
			if overwrite.upper() == 'Y':  
				ow_A = raw_input("OVERWRITE ACOUNT? (Y/N): ") 
				charge_density(root, model, AT, Q, CSIZE[i], nslice, nimage, a_type, force, nm, nxy, DIM, ow_A)
		else: charge_density(root, model, AT, Q, CSIZE[i], nslice, nimage, a_type, force, nm, nxy, DIM, 'Y')

model = 'TIP4P2005'
nsite, AT, Q, M, LJ = ut.get_param(model)
folder = 'SURFACE_2'
suffix = 'surface'
nfolder = 1
csize = 50
nsteps = 4000000
ntwr = 5000
ntwx = 1000
ntpr = 10
TYPE = 'W'
cutoff = 10
T = 298
nimage = 100

epsilon = LJ[0] * 4.184
sigma = np.max(LJ[1])
e_constant = 1.
st_constant = 1.
l_constant = 1E-10
rc = float(cutoff)

A_m = 1E-10
A_cm = 1E-8

print AT.index('lp'), AT.index('H')

root = "/data/fl7g13/AMBER/WATER/TIP4P2005/T_{}_K/CUT_{}_A/{}_TEST".format(T, cutoff, TYPE)
directory = '{}/{}_{}'.format(root, TYPE.upper(), 12)	
print '{}/{}/{}_{}_{}.nc'.format(directory, folder.upper(), model.lower(), csize, suffix)
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
nm = int(DIM[0] / sigma)
nxy = int((DIM[0]+DIM[1])/ sigma)

dx = DIM[0] / nxy
dy = DIM[1] / nxy
dz = DIM[2] / nslice

vcube = dx * dy * dz

C_D1 = np.zeros((nxy, nxy, nslice))
C_D2 = np.zeros((nxy, nxy, nslice))
C_D_z1 = np.zeros(nslice)
C_D_z2 = np.zeros(nslice)

Z = np.linspace(-DIM[2]/2., DIM[2]/2., nslice)

atom_types = list(set(AT))
n_atom_types = len(atom_types)

with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, nm, ntraj), 'r') as infile:
	int_av_density = np.loadtxt(infile)

c_den = np.array([int_av_density[1+atom_types.index('lp')][n] * Q[AT.index('lp')] + int_av_density[1+atom_types.index('H')][n] * Q[AT.index('H')] for n in range(nslice)]) * con.e / (vcube * 1E-30)

nimage = 100
for image in xrange(nimage):
	
	sys.stdout.write("LOADING {} out of {} CURVE files\r".format(image+1, nimage) )
	sys.stdout.flush()

	with file('{}/DATA/INTDEN/{}_{}_{}_{}_{}_CURVE.npz'.format(directory, model.lower(), csize, nm, nxy, image), 'r') as infile:
		npzfile = np.load(infile)
		XI1 = npzfile['XI1']
		XI2 = npzfile['XI2']

	for i in xrange(nxy):
		for j in xrange(nxy):
			for k in xrange(nslice):
				if Z[k] <= 0: dz = Z[k] - XI1[i][j]
				else: dz = XI2[i][j] - Z[k]
				m = int((dz+DIM[2]/2.) * nslice / DIM[2]) % nslice
				C_D1[i][j][k] += c_den[m] / nimage
				C_D_z1[k] += c_den[m] / (nxy**2 * nimage)

				
				dz = XI2[i][j] - Z[k]
				m = int((dz+DIM[2]/2.) * nslice / DIM[2]) % nslice
				C_D2[i][j][k] += c_den[m] / nimage
				C_D_z2[k] += c_den[m] / (nxy**2 * nimage)


plt.figure(0)
plt.plot(Z, C_D_z1)
#plt.plot(Z, C_D_z2)


F_T1 = np.fft.fftn(C_D_z1)
F_T_F1 = np.zeros((nslice))
F_T_P1 = np.zeros((nslice))

for i in xrange(nslice):
	if i != 0:
		F_T_F1[i] = F_T1[i] /  (np.pi * i)
		F_T_P1[i] = F_T1[i] / (np.pi**2 * i**2)

E_F1 = np.fft.ifftn(F_T_F1)
E_P1 = np.fft.ifftn(F_T_P1) / con.epsilon_0

plt.figure(1)
plt.plot(Z, E_F1)
plt.figure(2)
plt.plot(Z, E_P1)
plt.show()

e_field1 = spin.cumtrapz(C_D_z1, Z * 1E-10, initial = 0) / con.epsilon_0
e_field2 = spin.cumtrapz(C_D_z2, Z * 1E-10, initial = 0) / con.epsilon_0

plt.figure(1)
plt.plot(Z, e_field1)
plt.plot(Z, e_field2)

E_P1 = spin.cumtrapz(e_field1, Z * 1E-10, initial = 0)
E_P2 = - spin.cumtrapz(e_field2, Z * 1E-10, initial = 0)

plt.figure(2)
plt.plot(Z, E_P1)
plt.plot(Z, E_P2)
plt.show()

"""
nsite, AT, Q, M, LJ = ut.get_param('METHANOL')
nslice = 567
with file('/data/fl7g13/AMBER/METHANOL/T_298_K/CUT_18_A/METHANOL_50/SURFACE/DATA/INTDEN/methanol_50_567_14_2000_DEN.txt', 'r') as infile:
	int_av_mass_den, int_av_atom_den, int_av_mol_den, int_av_H_den, w_den_1, w_den_2 = np.loadtxt(infile)

c_den = [int_av_mol_den[n] * Q[AT.index('lp')] + int_av_H_den[n] * Q[AT.index('H')] for n in range(nslice)]

plt.plot(c_den)
plt.show()
"""
