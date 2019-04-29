"""
*************** MAIN INTERFACE MODULE *******************

Main program, ALIAS: Air-Liquid Interface Analysis Suite

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 27/2/2018 by Frank Longford


Parameters
----------

traj_file:  str
	Trajectory file name
top_file:  str
	Topology file name


"""
import numpy as np
import sys, os, subprocess
import mdtraj as md

import utilities as ut
import intrinsic_sampling_method as ism
import intrinsic_analysis as ia

def print_alias():

	print ' '+ '_' * 43
        print "|                   __ __             ____  |"
        print "|     /\     |        |       /\     /      |"
        print "|    /  \    |        |      /  \    \___   |"
        print "|   /___ \   |        |     /___ \       \  |"
        print "|  /      \  |____  __|__  /      \  ____/  |"
        print '|'+ '_' * 43 + '|' + '  v1.2.1'
        print ""
        print "    Air-Liquid Interface Analysis Suite"
        print ""


def run_alias(traj_file, top_file, recon=False, ow_coeff=False, ow_recon = False, ow_pos=False, ow_intpos=False, ow_hist=False, ow_dist=False):

	file_end = max([0] + [pos for pos, char in enumerate(traj_file) if char == '/'])
	traj_dir = traj_file[:file_end]
	traj_file = traj_file[file_end+1:]

	file_end = max([0] + [pos for pos, char in enumerate(top_file) if char == '/'])
	top_dir = top_file[:file_end]
	top_file = top_file[file_end+1:]

	alias_dir = traj_dir + '/alias_analysis/'
	data_dir = alias_dir + 'data/'
	figure_dir = alias_dir + 'figures/'
	if not os.path.exists(alias_dir): os.mkdir(alias_dir)
	if not os.path.exists(data_dir): os.mkdir(data_dir)
	if not os.path.exists(figure_dir): os.mkdir(figure_dir)

	print "Loading trajectory file {} using {} topology".format(traj_file, top_file)
	checkfile_name = alias_dir + traj_file.split('.')[0] + '_chk'
	if not os.path.exists('{}.pkl'.format(checkfile_name)):
		ut.make_checkfile(checkfile_name)

		traj, MOL = ut.get_sim_param('{}/{}'.format(traj_dir, traj_file), '{}/{}'.format(top_dir, top_file))

		#print "Simulation cell xyz dimensions in Angstoms: {}\n".format(dim)
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
		nframe = 0
		sys_M = [atom.element.mass for atom in traj.topology.atoms]
		N0 = [None, None]

		checkfile = ut.update_checkfile(checkfile_name, 'mol', mol)
		checkfile = ut.update_checkfile(checkfile_name, 'natom', natom)
		checkfile = ut.update_checkfile(checkfile_name, 'nmol', nmol)
		checkfile = ut.update_checkfile(checkfile_name, 'nsite', nsite)
		checkfile = ut.update_checkfile(checkfile_name, 'nframe', nframe)
		checkfile = ut.update_checkfile(checkfile_name, 'AT', AT)
		checkfile = ut.update_checkfile(checkfile_name, 'at_index', at_index)
		checkfile = ut.update_checkfile(checkfile_name, 'sys_M', sys_M)
		checkfile = ut.update_checkfile(checkfile_name, 'N0', N0)

	else:

		checkfile = ut.read_checkfile(checkfile_name)
		natom = checkfile['natom']
		nmol = checkfile['nmol']
		mol = checkfile['mol']
		nframe = checkfile['nframe']
		nsite = checkfile['nsite']
		sys_M = checkfile['sys_M']
		AT = checkfile['AT']
		at_index = checkfile['at_index']
		try: N0 = checkfile['N0']
		except:
			N0 = [0, 0]
			checkfile = ut.update_checkfile(checkfile_name, 'N0', N0)			

		print "Number of simulation frames: {}".format(nframe)
		print "Using residue {} for surface identification".format(mol)

	print "{} {} residues found, each containing {} atoms".format(nmol, mol, nsite)
	print "Atomic sites: {}".format(AT)

	if ('-M' in sys.argv): 
		mol_M = sys.argv[sys.argv.index('[') + 1 : sys.argv.index(']')]
		mol_M = [float(m) for m in mol_M]
	else:
		try:
			if bool(raw_input("\nUse elemental masses found in checkfile? {} g mol-1 (Y/N): ".format(checkfile['mol_M'])).upper() == 'Y'):
				mol_M = checkfile['mol_M']
			else: raise Exception
		except:
			if bool(raw_input("\nUse standard elemental masses? (Y/N): ").upper() == 'Y'):
				mol_M = [mass for mass in sys_M][:nsite]
			else:
				mol_M = np.zeros(nsite)
				for i in range(nsite):
					mol_M[i] = float(raw_input("   Enter mass for site {} g mol-1: ".format(AT[i])))
			print "Using atomic site masses: {} g mol-1".format(mol_M)
			checkfile = ut.update_checkfile(checkfile_name, 'mol_M', mol_M)
	
	print "Molar mass: {}  g mol-1".format(np.sum(mol_M))

	if ('-mol_com' in sys.argv): 
                mol_com = int(sys.argv[sys.argv.index('-mol_com') + 1])
                checkfile = ut.update_checkfile(checkfile_name, 'mol_com', mol_com)
        else:
		try:
			if bool(raw_input("\nUse centre of molecular mass in checkfile? {} (Y/N): ".format(checkfile['mol_com'])).upper() == 'Y'):
				mol_com = checkfile['mol_com']
			else: raise Exception
		except:
			if bool(raw_input("\nUse atomic sites as centre of molecular mass? (Y/N): ").upper() == 'Y'):
				sites = raw_input("   Site names: ").split()
				mol_com = [AT.index(site) for site in sites]
			else: mol_com = ['COM']
			checkfile = ut.update_checkfile(checkfile_name, 'mol_com', mol_com)

	file_name = "{}_{}_{}".format(traj_file.split('.')[0], mol, '_'.join([str(m) for m in mol_com]))

	if nframe == 0 or ow_pos:
		nframe = ut.make_mol_com('{}/{}'.format(traj_dir, traj_file), '{}/{}'.format(top_dir, top_file), data_dir, file_name, natom, nmol, AT, at_index, nsite, mol_M, sys_M, mol_com) 
		checkfile = ut.update_checkfile(checkfile_name, 'nframe', nframe)
		print "Number of simulation frames: {}".format(nframe)

		dim = ut.load_npy(data_dir + '/pos/{}_{}_dim'.format(file_name, nframe)).mean(axis=0)
		checkfile = ut.update_checkfile(checkfile_name, 'dim', dim)

	dim = checkfile['dim']
	print "Simulation cell xyz dimensions in Angstoms: {}\n".format(dim)

	if ('-mol_sigma' in sys.argv): 
                mol_sigma = float(sys.argv[sys.argv.index('-mol_sigma') + 1])
                checkfile = ut.update_checkfile(checkfile_name, 'mol_sigma', mol_sigma)
        else:
		try:
			if bool(raw_input("\nUse molecular radius found in checkfile? {} Angstroms (Y/N): ".format(checkfile['mol_sigma'])).upper() == 'Y'):
				mol_sigma = checkfile['mol_sigma']
			else: raise Exception
		except: 
			mol_sigma = float(raw_input("Enter molecular radius: (Angstroms) "))
			checkfile = ut.update_checkfile(checkfile_name, 'mol_sigma', mol_sigma)

	lslice = 0.05 * mol_sigma
	nslice = int(dim[2] / lslice)
	npi = 50

	q_max = 2 * np.pi / mol_sigma
	q_min = 2 * np.pi / np.sqrt(dim[0] * dim[1])
	qm = int(q_max / q_min)

	print "\n------STARTING INTRINSIC SAMPLING-------\n"
	print "Max wavelength = {:12.4f} sigma   Min wavelength = {:12.4f} sigma".format(q_max, q_min)
	print "Max frequency qm = {:6d}".format(qm)

	if ('-vlim' in sys.argv): 
                vlim = int(sys.argv[sys.argv.index('-vlim') + 1])
                checkfile = ut.update_checkfile(checkfile_name, 'vlim', vlim)
	else: vlim = 3

	if ('-ncube' in sys.argv): 
                ncube = int(sys.argv[sys.argv.index('-ncube') + 1])
                checkfile = ut.update_checkfile(checkfile_name, 'ncube', ncube)
	else: ncube = 3

	if ('-tau' in sys.argv): 
                vlim = int(sys.argv[sys.argv.index('-tau') + 1])
                checkfile = ut.update_checkfile(checkfile_name, 'tau', vlim)
	else: tau = 0.5

	if ('-max_r' in sys.argv): 
                vlim = int(sys.argv[sys.argv.index('-max_r') + 1])
                checkfile = ut.update_checkfile(checkfile_name, 'max_r', vlim)
	else: max_r = 1.5

	if ('-phi' in sys.argv): 
                phi = float(sys.argv[sys.argv.index('-phi') + 1])
                checkfile = ut.update_checkfile(checkfile_name, 'phi', phi)
        elif 'phi' in checkfile.keys():phi = checkfile['phi']
	else: 
		phi = 5E-8
		checkfile = ut.update_checkfile(checkfile_name, 'phi', phi)

	if ('-n0' in sys.argv):
                N0[recon] = int(sys.argv[sys.argv.index('-n0') + 1])
                checkfile = ut.update_checkfile(checkfile_name, 'N0', N0)
        else:
		try:
			if bool(raw_input("\nUse surface pivot number found in checkfile? {} pivots (Y/N): ".format(N0[recon])).upper() == 'Y'): pass
			else: raise Exception
		except:
			if bool(raw_input("\nManually enter in new surface pivot number? (search will commence otherwise): (Y/N)").upper() == 'Y'):
				N0[recon] = int(raw_input("\nEnter number of surface pivots: "))
			else:
				print "\n-------OPTIMISING SURFACE DENSITY-------\n"

				start_ns = 0.85
				ns, N0[recon] = ism.optimise_ns_diff(data_dir, file_name, nmol, nframe, qm, phi, dim, mol_sigma, start_ns, recon=recon,
												ncube=ncube, vlim=vlim, tau=tau, max_r=max_r)
			checkfile = ut.update_checkfile(checkfile_name, 'N0', N0)

	QM = range(1, qm+1)
	print "\nResolution parameters:"
	print "\n{:12s} | {:12s} | {:12s}".format('qu', "lambda (sigma)", "lambda (nm)")
	print "-" * 14 * 5 
	for qu in QM: print "{:12d} | {:12.4f} | {:12.4f}".format(qu, q_max / (qu*q_min), mol_sigma * q_max / (10*qu*q_min))
	print ""

	ism.create_intrinsic_surfaces(data_dir, file_name, dim, qm, N0[recon], phi, mol_sigma, nframe, recon=recon, ncube=ncube, vlim=vlim,
					 tau=tau, max_r=max_r, ow_coeff=ow_coeff, ow_recon=ow_recon)
	#ia.create_intrinsic_positions_dxdyz(data_dir, file_name, nmol, nframe, qm, N0[recon], phi, dim, recon=recon, ow_pos=ow_intpos)
	#ia.create_intrinsic_den_curve_hist(data_dir, file_name, qm, N0[recon], phi, nframe, nslice, dim, recon=recon, ow_hist=ow_hist)
	#ia.av_intrinsic_distributions(data_dir, file_name, dim, nslice, qm, N0[recon], phi, nframe, nframe, recon=recon, ow_dist=ow_dist)

	print"\n---- ENDING PROGRAM ----\n"

if __name__ == '__main__':
	print_alias()

	if len(sys.argv) < 2: traj_file = raw_input("Enter trajectory file: ")
	else: traj_file = sys.argv[1]
	while not os.path.exists(traj_file): traj_file = raw_input("\nTrajectory file not recognised: Re-enter file path: ")

	if len(sys.argv) < 3: top_file = raw_input("\nEnter topology file: ")
	else: top_file = sys.argv[2]
	while not os.path.exists(top_file): top_file = raw_input("\nTopology file not recognised: Re-enter file path: ")

	recon = ('-recon' in sys.argv)
        ow_coeff = ('-ow_coeff' in sys.argv)
        ow_recon = ('-ow_recon' in sys.argv)
	ow_pos = ('-ow_pos' in sys.argv)
        ow_intpos = ('-ow_intpos' in sys.argv)
        ow_hist = ('-ow_hist' in sys.argv)
        ow_dist = ('-ow_dist' in sys.argv or ow_hist)

	run_alias(traj_file, top_file, recon, ow_coeff, ow_recon, ow_pos, ow_intpos, ow_hist, ow_dist)
