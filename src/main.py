"""
*************** MAIN INTERFACE MODULE *******************

Main program, ALIAS: Air-Liquid Interface Analysis Suite

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 19/2/18 by Frank Longford


Parameters
----------

traj_file:  str
	Trajectory file name
top_file:  str
	Topology file name


"""
import numpy as np
import sys, os
import mdtraj as md

import utilities as ut
import intrinsic_sampling_method as ism

print ' '+ '_' * 43
print "|                   __ __             ____  |"
print "|     /\     |        |       /\     /      |" 
print "|    /  \    |        |      /  \    \___   |"
print "|   /___ \   |        |     /___ \       \  |"
print "|  /      \  |____  __|__  /      \  ____/  |"
print '|'+ '_' * 43 + '|' + '  v1.1'
print ""
print "    Air-Liquid Interface Analysis Suite"
print ""

if len(sys.argv) < 2: traj_file = raw_input("Enter trajectory file: ")
else: traj_file = sys.argv[1]
file_end = max([0] + [pos for pos, char in enumerate(traj_file) if char == '/'])
traj_dir = traj_file[:file_end]
traj_file = traj_file[file_end+1:]

if len(sys.argv) < 3: top_file = raw_input("\nEnter topology file: ")
else: top_file = sys.argv[2]
file_end = max([0] + [pos for pos, char in enumerate(top_file) if char == '/'])
top_dir = top_file[:file_end]
top_file = top_file[file_end+1:]

alias_dir = traj_dir + '/alias_analysis'
data_dir = alias_dir + '/data'
figure_dir = alias_dir + '/figures'
if not os.path.exists(alias_dir): os.mkdir(alias_dir)
if not os.path.exists(data_dir): os.mkdir(data_dir)
if not os.path.exists(figure_dir): os.mkdir(figure_dir)

print "Loading trajectory file {}".format(sys.argv[1])

traj, MOL, nframe, dim = ut.get_sim_param(traj_dir, top_dir, traj_file, top_file)

print "Number of simulation frames: {}".format(nframe)
print "Simulation cell xyz dimensions in Angstoms: {}\n".format(dim)
print "Residue types found: {}".format(len(MOL))
print "List of residues found: {}".format(MOL)
if len(MOL) > 1: mol = raw_input("\nChoose residue to use for surface identification: ")
else: 
	mol = MOL[0]
	print "Using residue {} for surface identification".format(mol)
atoms = [atom for atom in traj.topology.atoms if (atom.residue.name == mol)]
molecules = [molecule for molecule in traj.topology.residues if (molecule.name == mol)]

AT = [atom.name for atom in molecules[0].atoms]
at_index = [atom.index for atom in atoms]
nsite = molecules[0].n_atoms
natom = len(atoms)
nmol = len(molecules)

checkfile_name = '{}/{}_chk'.format(alias_dir, traj_file.split('.')[0])
if not os.path.exists('{}.pkl'.format(checkfile_name)): 
	ut.make_checkfile(checkfile_name)
	checkfile = ut.update_checkfile(checkfile_name, 'dim', dim)
	checkfile = ut.update_checkfile(checkfile_name, 'mol', mol)
	checkfile = ut.update_checkfile(checkfile_name, 'nmol', nmol)
	checkfile = ut.update_checkfile(checkfile_name, 'nframe', nframe)
checkfile = ut.read_checkfile(checkfile_name)

print "{} {} residues found, each containing {} atoms".format(nmol, mol, nsite)
print "Atomic sites: {}".format(AT)

try:
	if bool(raw_input("\nUse elemental masses found in checkfile? {} g mol-1 (Y/N): ".format(checkfile['M'])).upper() == 'Y'):
		M = checkfile['M']
	else: raise Exception
except:
	if bool(raw_input("\nUse standard elemental masses? (Y/N): ").upper() == 'Y'):
		M = [atom.element.mass for atom in traj.topology.atoms][:nsite]
	else:
		M = []
		for i in range(nsite):
			M.append(float(raw_input("   Enter mass for site {} g mol-1 ({}): ".format(i, AT[i]))))
		checkfile = ut.update_checkfile(checkfile_name, 'M', M)

print "Using atomic site masses: {} g mol-1".format(M)
print "Molar mass: {}  g mol-1".format(np.sum(M))

try:
	if bool(raw_input("\nUse centre of molecular mass in checkfile? {} (Y/N): ".format(checkfile['mol_com'])).upper() == 'Y'):
		mol_com = checkfile['mol_com']
	else: raise Exception
except:
	if bool(raw_input("\nUse atomic site as centre of molecular mass? (Y/N): ").upper() == 'Y'):
		mol_com = int(raw_input("   Site index: "))
	else: mol_com = 'COM'
	checkfile = ut.update_checkfile(checkfile_name, 'mol_com', mol_com)

file_name = "{}_{}_{}".format(traj_file.split('.')[0], mol, mol_com)
if not os.path.exists('{}/pos/{}_{}_com.npy'.format(data_dir, file_name, nframe)):
	ut.make_mol_com(traj, data_dir, file_name, natom, nmol, at_index, nframe, dim, nsite, M, mol_com) 

del traj

try:
	if bool(raw_input("\nUse molecular radius found in checkfile? {} Angstroms (Y/N): ".format(checkfile['mol_sigma'])).upper() == 'Y'):
		mol_sigma = checkfile['mol_sigma']
	else: raise Exception
except: 
	mol_sigma = float(raw_input("Enter molecular radius: (Angstroms) "))
	checkfile = ut.update_checkfile(checkfile_name, 'mol_sigma', mol_sigma)


try:
	if bool(raw_input("\nUse average temperature found in checkfile? {} K (Y/N): ".format(checkfile['T'])).upper() == 'Y'):
		T = checkfile['T']
	else: raise Exception
except: 
	T = float(raw_input("\nEnter average temperature of simulation in K: "))
	checkfile = ut.update_checkfile(checkfile_name, 'T', T)

lslice = 0.05 * mol_sigma
nslice = int(dim[2] / lslice)
npi = 50

q_max = 2 * np.pi / mol_sigma
q_min = 2 * np.pi / np.sqrt(dim[0] * dim[1])
qm = int(q_max / q_min)

print "\n------STARTING INTRINSIC SAMPLING-------\n"


print "Max wavelength = {:12.4f} sigma   Min wavelength = {:12.4f} sigma".format(q_max, q_min)
print "Max frequency qm = {:6d}".format(qm)

try:
	if bool(raw_input("\nUse weighting coefficient for surface area minimisation found in checkfile: phi = {}? (Y/N): ".format(checkfile['phi'])).upper() == 'Y'):
		phi = checkfile['phi']
	else: raise Exception
except: 
	if bool(raw_input("\nUse recommended weighting coefficient for surface area minimisation: phi = 5E-8? (Y/N): ").upper() == 'Y'):
		phi = 5E-8
	else: phi = float(raw_input("\nManually enter weighting coefficient: "))
	checkfile = ut.update_checkfile(checkfile_name, 'phi', phi)

try:
	if bool(raw_input("\nUse surface pivot number found in checkfile? {} pivots (Y/N): ".format(checkfile['n0'])).upper() == 'Y'):
		n0 = checkfile['n0']
	else: raise Exception
except:
	if bool(raw_input("\nManually enter in new surface pivot number? (search will commence otherwise): (Y/N)").upper() == 'Y'):
		n0 = int(raw_input("\nEnter number of surface pivots: "))
		checkfile = ut.update_checkfile(checkfile_name, 'n0', n0)
	else:
		print "\n-------OPTIMISING SURFACE DENSITY-------\n"

		start_ns = 0.75
		step_ns = 0.05

		print "Using initial pivot number = {}, step size = {}".format(int(dim[0] * dim[1] * start_ns / mol_sigma**2), int(dim[0] * dim[1] * step_ns / mol_sigma**2))

		ns, n0 = ism.optimise_ns(data_dir, file_name, nmol, nframe, qm, phi, dim, mol_sigma, start_ns, step_ns)
		checkfile = ut.update_checkfile(checkfile_name, 'n0', n0)

QM = range(1, qm+1)
print "\nResolution parameters:"
print "\n{:12s} | {:12s} | {:12s}".format('qu', "lambda (sigma)", "lambda (nm)")
print "-" * 14 * 5 
for qu in QM: print "{:12d} | {:12.4f} | {:12.4f}".format(qu, q_max / (qu*q_min), mol_sigma * q_max / (10*qu*q_min))
print ""

ow_coeff = bool(raw_input("OVERWRITE ACOEFF? (Y/N): ").upper() == 'Y')
ow_recon = bool(raw_input("OVERWRITE RECON ACOEFF? (Y/N): ").upper() == 'Y')
ow_pos = bool(raw_input("OVERWRITE POSITIONS? (Y/N): ").upper() == 'Y')
ow_count = bool(raw_input("OVERWRITE DENSITY COUNT? (Y/N): ").upper() == 'Y')

ism.create_intrinsic_surfaces(data_dir, file_name, dim, qm, n0, phi, mol_sigma, nframe, recon=True, ow_coeff=ow_coeff, ow_recon=ow_recon)

for recon in [False, True]:
	ism.create_intrinsic_positions_dxdyz(data_dir, file_name, nmol, nframe, qm, n0, phi, dim, recon=recon, ow_pos=ow_pos)
	ism.create_intrinsic_den_curve_dist(data_dir, file_name, qm, n0, phi, nframe, nslice, dim, recon=recon, ow_count=ow_count)
