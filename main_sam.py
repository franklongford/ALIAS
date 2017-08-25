"""
*************** MAIN INTERFACE MODULE *******************

Main program, ALIAS: Air-Liquid Interface Analysis Suite

Parameter definitions:

	model:    Forcefield being used
	nsite:	  Number of atomic sites

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 24/11/2016 by Frank Longford
"""

import numpy as np
import os


import utilities as ut
import sys
import subprocess


	

print ' '+ '_' * 43
print "|                   __ __             ____  |"
print "|     /\     |        |       /\     /      |" 
print "|    /  \    |        |      /  \    \___   |"
print "|   /___ \   |        |     /___ \       \  |"
print "|  /      \  |____  __|__  /      \  ____/  |"
print '|'+ '_' * 43 + '|' + '  v0.2'
print ""
print "    Air-Liquid Interface Analysis Suite"
print ""



model = True

"Model input from user"
while model:
	model = raw_input("Which model?\n\nArgon\nSPCE\nTIP3P\nTIP4P2005\nAMOEBA\nMethanol\nEthanol\nDMSO\n\n")
       	model = model.upper()
	if model in ['ARGON', 'SPCE', 'TIP3P', 'TIP4P2005', 'AMOEBA', 'METHANOL', 'ETHANOL', 'DMSO']:
		"Model must be known already" 
		print "Using {}".format(model)
		"Parameters to be input by user"
		T = int(raw_input("Temperature: (K) "))
		cutoff = int(raw_input("Cutoff: (A) "))
		func = raw_input("Function:\nTest or Slab? (T, S): ")
		break

	else:
		"Otherwise choose again"
		print 'Model unrecognised, try again'
		model = True




nsite, AT, Q, M, LJ, mol_sigma = ut.get_param(model)

"Directory information is found in"
if model in ['METHANOL', 'ETHANOL', 'DMSO', 'AMOEBA']: folder = 'SURFACE'
else: folder = 'SURFACE_2'

suffix = 'surface'

if model in ['METHANOL', 'ETHANOL', 'DMSO']:
	a_type = 'calc'
	com = 'COM'
else: 
	com = '0'
	if model == 'AMOEBA': a_type = 'ame'
	else: a_type = 'exp'

if model in ['ARGON', 'METHANOL', 'ETHANOL', 'DMSO']: root = '/local/scratch/sam5g13/AMBER/{}/T_{}_K/CUT_{}_A'.format(model, T, cutoff)
elif model == 'AMOEBA': root = '/local/scratch/sam5g13/OpenMM/WATER/{}/T_{}_K/CUT_{}_A'.format(model, T, cutoff)
else: root = '/local/scratch/sam5g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A'.format(model, T, cutoff)

if not os.path.exists(root):
	print "FILES NOT FOUND"
	sys.exit()


if func.upper() == 'S':
	"---------------------------------------------ENTERING SLAB-----------------------------------------------------"

	import sys
	import mdtraj as md
	from scipy import constants as con
	import matplotlib.pyplot as plt

	import density as den
	import intrinsic_surface as surf
	import orientational as ori
	import dielectric as die
	import ellipsometry as ellips
	import graphs
	

	TYPE = 'SLAB'

	if model == 'AMOEBA': csize = 50
	elif model == 'DMSO': csize = 120
	else: csize = 50

	root = '{}/{}'.format(root, TYPE.upper())

	sigma = np.max(LJ[1])

	if model not in ['AMOEBA']:

		rad_dist = bool(raw_input("PERFORM RADIAL DISTRIBUTION? (Y/N): ").upper() == 'Y')

		if rad_dist:
			print "\n-------------CUBIC RADIAL DISTRIBUTION-----------\n"
	
			directory = '{}/CUBE'.format(root)

			if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))

			if not os.path.exists("{}/DATA/DEN".format(directory)): os.mkdir("{}/DATA/DEN".format(directory))

			traj = ut.load_nc(root, 'CUBE', model, csize, 'cube')
			

			"""
			"Original code"
			traj = ut.load_nc(root, 'CUBE', model, csize, 'cube')
			print traj
			"""

			natom = int(traj.n_atoms)
			nmol = int(traj.n_residues)
			nframe = int(traj.n_frames)
			"lOADING DIMENSIONS IN NM, CONVERT TO ANGTROMS"
			DIM = np.array(traj.unitcell_lengths[0]) * 10

			lslice = 0.01
			lslice_nm = lslice / 10.
			max_r = np.min(DIM) / 20.
			nslice = int(max_r / lslice)
			
			
			new_XYZ = np.zeros((nframe, natom, 3))
			for image in xrange(nframe):
		
			
				XYZ= np.transpose(traj.xyz[image])	
				
				xmol, ymol, zmol = ut.molecules(XYZ[0], XYZ[1], XYZ[2], nsite, M, com="COM")

				XYZ = np.transpose([xmol, ymol, zmol])
				for i in xrange(nmol): new_XYZ[image, i] = XYZ[i]
			traj.xyz = new_XYZ
			
			pairs = []
			for i in xrange(100): 
				for j in xrange(i): pairs.append([i, j])
				#for j in xrange(i): pairs.append([i*nsite, j*nsite])
			#print pairs

			r, g_r = md.compute_rdf(traj, pairs = pairs, bin_width = lslice_nm, r_range = (0, max_r))
			plt.plot(r*10, g_r)
			
			
			with open('{}/DATA/DEN/{}_{}_{}_RDEN.txt'.format(directory, model.lower(), nslice, nframe), 'w') as outfile:
				np.savetxt(outfile, (r, g_r), fmt='%-12.6f')

			mol_sigma = 2**(1./6) * g_r.argmax() * lslice

			print "r_max = {}    molecular sigma = {}".format(g_r.argmax()*lslice, mol_sigma)
			plt.show()

			
	

	"END OF RDF"

	directory = '{}/{}'.format(root, folder.upper())
	

	print "\n----------BUILDING SURFACE POSITIONAL ARRAYS-----------\n"

	if not os.path.exists("{}/DATA/POS".format(directory)): os.mkdir("{}/DATA/POS".format(directory))

	ow_pos = bool(raw_input("OVERWRITE AT MOL POSITIONS? (Y/N): ").upper() == 'Y')

	if os.path.exists('{}/DATA/parameters.txt'.format(directory)) and not ow_pos:
		DIM = np.zeros(3)
		with file('{}/DATA/parameters.txt'.format(directory), 'r') as infile:
			natom, nmol, nframe, DIM[0], DIM[1], DIM[2] = np.loadtxt(infile)
		natom = int(natom)
		nmol = int(nmol)
		nframe = int(nframe)

		print 'LOADING PARAMETER AND COM FILES'
		with file('{}/DATA/POS/{}_{}_COM.txt'.format(directory, model.lower(), nframe), 'r') as infile:
			cell_com = np.loadtxt(infile)

	else:
		print '{}/{}/{}_{}_{}.nc'.format(root, folder.upper(), model.lower(), csize, suffix)

		traj = ut.load_nc(root, directory, model, suffix)
	
		ut.at_mol_positions(root, directory, model, nframe, natom,  suffix, nsite, M)	
	
		
	lslice = 0.05 * sigma
	nslice = int(DIM[2] / lslice)
	vlim = 3
	ncube = 3
	
	"""mol sigma should be determined by the radial distribution"""

	if model in ['TIP4P2005', 'ARGON', 'AMOEBA', 'SPCE']: mol_sigma = sigma
	elif model == 'METHANOL': mol_sigma = 3.83
	elif model == 'ETHANOL': mol_sigma = 4.57
	elif model == 'DMSO': mol_sigma = 5.72

	"""2nm + 1 = the number of waves that goes into the intrinsic surface"""
	nm = int((DIM[0] + DIM[1]) / (2 * mol_sigma))

	
	print "\n-----------STARTING DENSITY PROFILE------------\n"
	
	if not os.path.exists("{}/DATA/DEN".format(directory)): os.mkdir("{}/DATA/DEN".format(directory))

	ow_all = False
	ow_count = False

	if os.path.exists('{}/DATA/DEN/{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nframe)) and not ow_all:
		print "FILE FOUND '{}/DATA/DEN/{}_{}_{}_DEN.txt".format(directory, model.lower(), nslice, nframe)
		if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):
			ow_count = bool(raw_input("OVERWRITE COUNT? (Y/N): ").upper() == 'Y')
			den.density_profile(directory, model, nframe, natom, nmol, nsite, AT, M, cell_com, DIM, nslice, ow_count)	
	else: den.density_profile(directory, model, nframe, natom, nmol, nsite, AT, M, cell_com, DIM, nslice, ow_all)
	
	print "\n------STARTING INTRINSIC DENSITY PROFILE-------\n"

	ow_all = False
	ow_coeff = False
	ow_curve = False
	ow_count = False
	ow_wden = False

	if not os.path.exists("{}/DATA/INTDEN".format(directory)): os.mkdir("{}/DATA/INTDEN".format(directory))
	if not os.path.exists("{}/DATA/INTPOS".format(directory)): os.mkdir("{}/DATA/INTPOS".format(directory))

	if os.path.exists('{}/DATA/INTDEN/{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nm, nframe)) and not ow_all:
		print "FILE FOUND '{}/DATA/INTDEN/{}_{}_{}_{}_DEN.txt".format(directory, model.lower(), nslice, nm, nframe)
		if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):
			ow_coeff = bool(raw_input("OVERWRITE ACOEFF? (Y/N): ").upper() == 'Y')
			ow_count = bool(raw_input("OVERWRITE COUNT? (Y/N): ").upper() == 'Y')
			ow_wden = bool(raw_input("OVERWRITE WDEN? (Y/N): ").upper() == 'Y')		
			surf.intrinsic_profile(directory, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, mol_sigma, cell_com, DIM, nslice, ncube, nm, vlim, ow_coeff, ow_curve, ow_count, ow_wden)
	else: surf.intrinsic_profile(directory, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, mol_sigma, cell_com, DIM, nslice, ncube, nm, vlim, ow_all, ow_all, ow_all, ow_all)

	#graphs.print_graphs_density(directory, model, nsite, AT, nslice, nm, cutoff, csize, folder, suffix, nframe, DIM)

	if model != 'ARGON':

		ow_all = False
		ow_angles = False

		print "\n--------STARTING ORIENTATIONAL PROFILE--------\n"

		if not os.path.exists("{}/DATA/EULER".format(directory)): os.mkdir("{}/DATA/EULER".format(directory))
		if not os.path.exists("{}/DATA/INTEULER".format(directory)): os.mkdir("{}/DATA/INTEULER".format(directory))

		if os.path.exists('{}/DATA/EULER/{}_{}_{}_{}_EUL.txt'.format(directory, model.lower(), nslice, a_type, nframe)) and not ow_all:
			print 'FILE FOUND {}/DATA/EULER/{}_{}_{}_{}_EUL.txt'.format(directory, model.lower(), nslice, a_type, nframe)
			if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):  
				ow_angles = bool(raw_input("OVERWRITE ANGLES? (Y/N): ").upper() == 'Y')
				ow_polar = bool(raw_input("OVERWRITE POLARISABILITY? (Y/N): ").upper() == 'Y') 
				ori.euler_profile(directory, nframe, nslice, nmol, model, csize, suffix, AT, Q, M, LJ, cell_com, DIM, nsite, a_type, nm, ow_angles, ow_polar)
		else: ori.euler_profile(directory, nframe, nslice, nmol, model, csize, suffix, AT, Q, M, LJ, cell_com, DIM, nsite, a_type, nm, ow_all, ow_all)

	#graphs.print_graphs_orientational(directory, model, nsite, AT, nslice, nm, a_type, cutoff, csize, folder, suffix, nframe, DIM)

	ow_all = False
	ow_ecount = False
	ow_acount = False

	print "\n-------STARTING DIELECTRIC PROFILE--------\n"

	if not os.path.exists("{}/DATA/DIELEC".format(directory)): os.mkdir("{}/DATA/DIELEC".format(directory))
	if not os.path.exists("{}/DATA/INTDIELEC".format(directory)): os.mkdir("{}/DATA/INTDIELEC".format(directory))
	if not os.path.exists("{}/DATA/ELLIP".format(directory)): os.mkdir("{}/DATA/ELLIP".format(directory))

	if os.path.exists('{}/DATA/DIELEC/{}_{}_{}_{}_DIE.txt'.format(directory, model.lower(), nslice, a_type, nframe)) and not ow_all:
		print 'FILE FOUND {}/DATA/DIELEC/{}_{}_{}_{}_DIE.txt'.format(directory, model.lower(), nslice, a_type, nframe)
		if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):   
			#ow_ecount = bool(raw_input("OVERWRITE ECOUNT? (Y/N): ").upper() == 'Y')
			#ow_acount = bool(raw_input("OVERWRITE ACOUNT? (Y/N): ").upper() == 'Y') 
			die.dielectric_refractive_index(directory, model, csize, AT, sigma, nslice, nframe, a_type, nm, DIM, ow_ecount, ow_acount)
	else: die.dielectric_refractive_index(directory, model, csize, AT, sigma, nslice, nframe, a_type, nm, DIM, ow_all, ow_acount)

	#graphs.print_graphs_dielectric(directory, model, nsite, AT, nslice, nm, a_type, cutoff, csize, folder, suffix, nframe, DIM)

	print "\n-------STARTING ELLIPSOMETRY PREDICTIONS--------\n"

	ellips.transfer_matrix(directory, model, csize, AT, sigma, nslice, nframe, a_type, nm, DIM, cutoff)


elif func.upper() == 'T':
	"---------------------------------------------ENTERING TEST-----------------------------------------------------"

	TYPE = raw_input("Width (W), Area (A) or Cubic (C) variation: ")
	force = raw_input("VDW Force corrections? (Y/N): ")

	root = '{}/{}_TEST'.format(root, TYPE.upper())

	if force.upper() == 'Y': folder = 'SURFACE_2'
	else: folder = 'SURFACE' 

	suffix = 'surface'
	csize = 50

	if model == 'ARGON':
		if folder.upper() == 'SURFACE_2':
			if TYPE.upper() == 'W': 
				nfolder = 60
				sfolder = 11 
			elif TYPE.upper() == 'A': 
				nfolder = 25
				sfolder = 4
			elif TYPE.upper() == 'C': 
				nfolder = 22
				sfolder = 0
		else:
			if TYPE.upper() == 'W': 
				nfolder = 60
				sfolder = 11
			elif TYPE.upper() == 'C': 
				nfolder = 7
				csize = 30

	if model == 'TIP4P2005':
		if TYPE.upper() == 'W': 
			nfolder = 40
			sfolder = 11 
		if TYPE.upper() == 'A': nfolder = 25
		if TYPE.upper() == 'C':
		        nfolder = 2
		        csize = 35
		if T != 298: nfolder = 1	

	if model in ['SPCE', 'TIP3P']:
		if TYPE.upper() == 'W':
			nfolder = 40
			sfolder = 11

	print ""
	build = bool(raw_input("Make input files or Analyse?").upper() == 'Y')

	if build: pass
	else:

		TASK = raw_input("What task to perform?\nD  = Density Profile\nIS = Intrinsic Surface Profiling\nO  = Orientational Profile\nE  = Dielectric and Refractive Index.\nT  = Thermodynamics\nEL = Ellipsometry module\nG  = Print Graphs\n")
		print ""

		if TASK.upper() == 'D':

			ow_all =  bool(raw_input("OVERWRITE ALL DENISTY? (Y/N): ").upper() == 'Y')
			sigma = np.max(LJ[1])
			lslice = 0.05 * sigma

			for i in xrange(sfolder, nfolder):
				root_ = '{}/{}_{}'.format(root, TYPE.upper(), i)
				directory = '{}/{}'.format(root_, folder.upper())
		
				ow_den = True
				ow_count = False

				if os.path.exists('{}/DATA/parameters.txt'.format(directory)) and not ow_all:
					print "LOADING {}/DATA/parameters.txt".format(directory)
					with file('{}/DATA/parameters.txt'.format(directory), 'r') as infile:
						_, _, nframe, _, _, dim_Z = np.loadtxt(infile)

					nframe = int(nframe)
					nslice = int(dim_Z / lslice)

					if os.path.exists('{}/DATA/DEN/{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nframe)) and not ow_all:
						print "FILE FOUND '{}/DATA/DEN/{}_{}_{}_DEN.txt".format(directory, model.lower(), nslice, nframe)
						ow_den = bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y')
						if ow_den: ow_count = bool(raw_input("OVERWRITE COUNT? (Y/N): ").upper() == 'Y')

				if ow_den or ow_all:
					import density as den

					print '{}/{}/{}_{}.nc'.format(root_, folder.upper(), model.lower(),suffix)
					traj = ut.load_nc(root_, folder, model, suffix)

					if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))

					natom = int(traj.n_atoms)
					nmol = int(traj.n_residues)
					nframe = int(traj.n_frames)
					DIM = np.array(traj.unitcell_lengths[0]) * 10

					with file('{}/DATA/parameters.txt'.format(directory), 'w') as outfile:
						np.savetxt(outfile, [natom, nmol, nframe, DIM[0], DIM[1], DIM[2]])

					nslice = int(DIM[2] / lslice)

					if not os.path.exists("{}/DATA/DEN".format(directory)): os.mkdir("{}/DATA/DEN".format(directory))
						
					if ow_all: den.density_thermo(traj, directory, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, com, DIM, nslice, ow_all)
					else: den.density_thermo(traj, directory, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, com, DIM, nslice, ow_count)

				print ""

		elif TASK.upper() == 'T':

			import thermodynamics as thermo

			rc = float(cutoff)

			"Conversion of length and surface tension units"
			if model == 'ARGON':
				LJ[0] = LJ[0] * 4.184
				e_constant = 1 / LJ[0]
				st_constant = ((LJ[1]*1E-10)**2) * con.N_A * 1E-6 / LJ[0]
				l_constant = 1 / LJ[1]
				T = 85
				com = 0
			else: 
				LJ[0] = LJ[0] * 4.184
				e_constant = 1.
				st_constant = 1.
				l_constant = 1E-10
				T = 298

			ow_area = bool(raw_input("OVERWRITE INTRINSIC SURFACE AREA? (Y/N): ").upper() == 'Y')
			ow_ntb = bool(raw_input("OVERWRITE SURFACE TENSION ERROR? (Y/N): ").upper() == 'Y')
			ow_est = bool(raw_input("OVERWRITE AVERAGE ENERGY AND TENSION? (Y/N): ").upper() == 'Y')
			(ENERGY, ENERGY_ERR, TEMP, TEMP_ERR, TENSION, TENSION_ERR, VAR_TENSION, N_RANGE, A_RANGE, AN_RANGE, Z_RANGE, DEN) = thermo.energy_tension(
				root, model, suffix, TYPE, folder, sfolder, nfolder, T, rc, LJ, csize, e_constant, l_constant, st_constant, com, ow_area, ow_ntb, ow_est)

		elif TASK.upper() == 'G':
			import graphs
			graphs.print_graphs_thermodynamics(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, folder, suffix)

