"""
*************** TEST MODULE *******************



***************************************************************
Created 19/07/2017 by Sam Munday

Contributors: Sam Munday

Last modified 19/07/2017 by Sam Munday
"""

import numpy as np
import scipy as sp
import time, sys, os
from scipy import constants as con
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import utilities as ut
import intrinsic_surface as int_surf
import density as den
import mdtraj as md

root = '/local/scratch/sam5g13/AMBER/TEST'
model = 'SPCE'
T = 298 
cutoff = 10

nsite, AT, Q, M, LJ, mol_sigma = ut.get_param(model)

def test_params():
	"If assert is false we know we've changed module"

	assert nsite == 3
	assert np.sum(Q) == 0
	assert len(set(AT)) == 2
	assert np.sum(M) == 18.016
	assert np.max(LJ[1]) == 3.166	


def test_rdf():
	
	mol_sigma_rdf, r_max = den.radial_dist(root, model, nsite, M, '0')

	assert mol_sigma_rdf == 3.0755460123676825

def test_at_mol():

	directory = '{}/SURFACE_2'.format(root)
	suffix = 'surface'
	traj = md.load_frame('{}/{}_{}.nc'.format(directory, model.lower(), suffix), 0, top='{}/{}.prmtop'.format(root, model.lower()))

	XYZ = np.transpose(traj.xyz[0])
	XAT_traj = XYZ[0] * 10
	YAT_traj = XYZ[1] * 10
	ZAT_traj = XYZ[2] * 10
	
	XMOL_traj, YMOL_traj, ZMOL_traj = ut.molecules(XAT_traj, YAT_traj, ZAT_traj, nsite, M, com = 'COM')

	ut.save_atom_positions(XAT_traj, YAT_traj, ZAT_traj, directory, model, 0)
	ut.save_mol_positions(XMOL_traj, YMOL_traj, ZMOL_traj, directory, model, 0)

	xat, yat, zat = ut.read_atom_positions(directory, model, 0)
	xmol, ymol, zmol = ut.read_mol_positions(directory, model, 0)
	
	
	assert np.sum(XAT_traj - xat) <= 2E-5
	assert np.sum(YAT_traj - yat) <= 2E-5
	assert np.sum(ZAT_traj - zat) <= 2E-5

	assert np.sum(XMOL_traj - xmol) <= 2E-5
	assert np.sum(YMOL_traj - ymol) <= 2E-5
	assert np.sum(ZMOL_traj - zmol) <= 2E-5

def test_com():
	
	directory = '{}/SURFACE_2'.format(root)
	xat, yat, zat = ut.read_atom_positions(directory, model, 0)
	xcom, ycom, zcom = ut.centre_mass(xat, yat, zat, nsite, M)

	com_0 = np.array([17.627918,  17.423219,  36.004216])
	com_test = np.round(np.array([xcom, ycom, zcom], dtype = float), decimals = 6)

	assert np.sum(com_0 - com_test) == 0.


def test_at_mol_traj():

	directory = '{}/SURFACE_2'.format(root)
	suffix = 'surface'
	traj = md.load_frame('{}/{}_{}.nc'.format(directory, model.lower(), suffix), 0, top='{}/{}.prmtop'.format(root, model.lower()))

	assert ut.at_mol_positions(traj, directory, model, nsite, M, 0)

	
def test_intrinsic_surface():

	directory = '{}/SURFACE_2'.format(root)
	suffix = 'surface'
	traj = md.load_frame('{}/{}_{}.nc'.format(directory, model.lower(), suffix), 0, top='{}/{}.prmtop'.format(root, model.lower()))

	natom = int(traj.n_atoms)
	nmol = int(traj.n_residues)
	DIM = np.array(traj.unitcell_lengths[0]) * 10

	XYZ = np.transpose(traj.xyz[0])
	xat = XYZ[0] * 10
	yat = XYZ[1] * 10
	zat = XYZ[2] * 10
	
	xmol, ymol, zmol = ut.molecules(xat, yat, zat, nsite, M, com = '0')
	xcom, ycom, zcom = ut.centre_mass(xat, yat, zat, nsite, M)

	phi = 5E-2
	c = 1.3
	max_r = 1.5 * mol_sigma
	tau = 0.4 * mol_sigma
	nm = int((DIM[0] + DIM[1]) / (2 * mol_sigma))
	n0 = int(DIM[0] * DIM[1] * c / mol_sigma**2)
	ncube = 3
	vlim = 3

	auv1, auv2, piv_n1, piv_n2 = int_surf.build_surface(xmol, ymol, zmol, DIM, nmol, ncube, mol_sigma, nm, n0, vlim, phi, zcom, tau, max_r)

test_intrinsic_surface()	

	
