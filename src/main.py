"""
*************** MAIN INTERFACE MODULE *******************

Main program, ALIAS: Air-Liquid Interface Analysis Suite

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 24/11/2016 by Frank Longford
"""
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as con
from scipy import stats
from scipy.optimize import curve_fit
import os

import utilities as ut
import mdtraj as md

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

model = raw_input("Which model?\n\nArgon\nSPCE\nTIP3P\nTIP4P2005\nAMOEBA\nMethanol\nEthanol\nDMSO\n\n")

nsite, AT, Q, M, LJ = ut.get_model_param(model)

if model.upper() in ['METHANOL', 'ETHANOL', 'DMSO', 'AMOEBA']: folder = 'SURFACE'
else: folder = 'SURFACE_2'

suffix = 'surface'

if model.upper() in ['METHANOL', 'ETHANOL', 'DMSO']:
	a_type = 'calc'
	com = 'COM'
else: 
	com = '0'
	if model.upper() == 'AMOEBA': a_type = 'ame'
	else: a_type = 'exp'

T = int(raw_input("Temperature: (K) "))
rc = int(raw_input("Cutoff: (A) "))
func = raw_input("Function:\nThermodynamics or Ellipsometry? (T, E): ")

if model.upper() in['ARGON', 'METHANOL', 'ETHANOL', 'DMSO']: root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A'.format(model.upper(), T, rc)
else: root = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A'.format(model.upper(), T, rc)

if func.upper() == 'T':
	import density as den
	import thermodynamics as thermo
	import graphs

	#TYPE = raw_input("Width (W), Area (A) or Cubic (C) variation: ")
	force = raw_input("VDW Force corrections? (Y/N): ")

	if force.upper() == 'Y': folder = 'SURFACE_2'
	else: folder = 'SURFACE' 

	suffix = 'surface'
	csize = 50

	red_units = False
	conv_mJ_kcalmol = 1E-6 / 4.184 * con.N_A

	LJ[0] = LJ[0] * 4.184
	e_constant, st_constant, l_constant, p_constant, T_constant = ut.get_thermo_constants(red_units, LJ)

	nbin = 300

	"Conversion of length and surface tension units"
        if model.upper() == 'ARGON':
                gam_start = -300 * st_constant
                gam_end = 300 * st_constant
                com = 0
		TEST = ['W', 'A', 'C']
		if folder.upper() == 'SURFACE': NFOLDER = [49, 0, 0]
        	elif folder.upper() == 'SURFACE_2': NFOLDER = [29, 21, 12]
        	COLOUR = ['b', 'r', 'g']
        	CSIZE = [50, 50, 50]
		root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A'.format(model.upper(), T, rc)
        else:
                gam_start = -500 * st_constant
                gam_end = 500 * st_constant
               
		TEST = ['W']#, 'A', 'C']
		NFOLDER = [29, 0, 0]
		COLOUR = ['b', 'r', 'g']
		CSIZE = [50, 50, 50]

		if model.upper() in ['METHANOL', 'ETHANOL', 'DMSO']: 
			com = 'COM'
			root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A'.format(model.upper(), T, rc)
                else: 
			com = 0
			root = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A'.format(model.upper(), T, rc)

	#TASK = raw_input("What task to perform?\nD  = Density Profile\nIS = Intrinsic Surface Profiling\nO  = Orientational Profile\nE  = Dielectric and Refractive Index.\nT  = Thermodynamics\nEL = Ellipsometry module\nG  = Print Graphs\n")

	TOT_ENERGY = []
	TOT_ENERGY_ERR = []
	TOT_TEMP = []
	TOT_TEMP_ERR = []
	TOT_TENSION = []
	TOT_TENSION_ERR = []

	TOT_VAR_TENSION = []
	TOT_Z_RANGE = []
	TOT_A_RANGE = []
	TOT_N_RANGE = []
	
	TOT_AN_RANGE = []
	TOT_DEN_RANGE = []
	TOT_OMEGA_RANGE = []

	for i, TYPE in enumerate(TEST):
		csize = CSIZE[i]
		nfolder = NFOLDER[i]

		ow_den =  bool(raw_input("OVERWRITE ALL DENSITY? (Y/N): ").upper() == 'Y')
		ow_thermo =  bool(raw_input("OVERWRITE ALL THERMODYNAMICS? (Y/N): ").upper() == 'Y')

		sigma = np.max(LJ[1])
		epsilon = np.max(LJ[0])
		lslice = 0.05 * sigma

		ENERGY = np.zeros(nfolder)
		ENERGY_ERR = np.zeros(nfolder)
		TEMP = np.zeros(nfolder)
		TEMP_ERR = np.zeros(nfolder)
		TENSION = np.zeros(nfolder)
		TENSION_ERR = np.zeros(nfolder)

		VAR_TENSION = np.zeros(nfolder)
		N_RANGE = np.zeros(nfolder)
		A_RANGE = np.zeros(nfolder)
		Z_RANGE = np.zeros(nfolder)
		AN_RANGE = np.zeros(nfolder)

		DEN_RANGE = np.zeros(nfolder)
		OMEGA_RANGE = np.zeros(nfolder)

		for n in xrange(nfolder):
			root_ = '{}/{}_TEST/{}_{}'.format(root, TYPE.upper(), TYPE.upper(), n)
			directory = '{}/{}'.format(root_, folder.upper())
			data_dir = "{}/DATA".format(directory)

			if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))

			natom, nmol, ntraj, DIM, COM = ut.get_sim_param(root_, directory, data_dir, model, nsite, suffix, csize, M, com, False, False)

			nslice = int(DIM[2] / lslice)
			nframe = ntraj

			A_RANGE[n] = 2 * DIM[0] * DIM[1] * l_constant**2	
			N_RANGE[n] = nmol
			if red_units == False: AN_RANGE[n] = A_RANGE[n] * con.N_A / nmol
			else: AN_RANGE[n] = A_RANGE[n] / nmol

			print "\n-----------STARTING DENSITY PROFILE------------\n"
	
			if not os.path.exists("{}/DATA/DEN".format(directory)): os.mkdir("{}/DATA/DEN".format(directory))

			ow_count = False

			if os.path.exists('{}/DATA/DEN/{}_{}_{}_DEN.npy'.format(directory, model.lower(), nslice, nframe)) and not ow_den:
				print "FILE FOUND  {}/DATA/DEN/{}_{}_{}_DEN.npy".format(directory, model.lower(), nslice, nframe)
				#if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):
				#	ow_count = bool(raw_input("OVERWRITE COUNT? (Y/N): ").upper() == 'Y')
				#	den.density_profile(directory, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, COM, DIM, nslice, ow_count)	
			else: den.density_profile(data_dir, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, COM, DIM, nslice, ow_den)

			with file('{}/DATA/DEN/{}_{}_{}_PAR.npy'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
				param = np.load(infile)

			Z_RANGE[n] = 2 * param[3] * l_constant
			DEN_RANGE[n] = param[0]
			OMEGA_RANGE[n] =  param[-1] * 2.1972 * l_constant

			print "\n-----------STARTING THERMODYNAMIC ANALYSIS------------\n"

			ow_ntb = False

			if not os.path.exists("{}/DATA/THERMO".format(directory)): os.mkdir("{}/DATA/THERMO".format(directory))
			if os.path.exists('{}/DATA/THERMO/{}_{}_E_ST.txt'.format(directory, model.lower(), nframe)): ut.convert_txt_npy('{}/DATA/THERMO/{}_{}_E_ST'.format(directory, model.lower(), nframe))
			if os.path.exists('{}/DATA/THERMO/{}_{}_TOT_E_ST.txt'.format(directory, model.lower(), nframe)): ut.convert_txt_npy('{}/DATA/THERMO/{}_{}_TOT_E_ST'.format(directory, model.lower(), nframe))

			if os.path.exists('{}/DATA/THERMO/{}_{}_E_ST.npy'.format(directory, model.lower(), nframe)) and not ow_thermo:
				print 'FILE FOUND  {}/DATA/THERMO/{}_{}_E_ST.npy'.format(directory, model.lower(), nframe)
				#if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):
				#	ow_ntb = bool(raw_input("OVERWRITE BLOCK ERRORS? (Y/N): ").upper() == 'Y')
				#	thermo.get_thermo(directory, model, csize, suffix, nslice, nframe, DIM, nmol, rc, sigma, epsilon, ow_ntb)
			else: thermo.get_thermo(directory, model, csize, suffix, nslice, nframe, DIM, nmol, rc, sigma, epsilon, ow_thermo, 'J')
			
			with file('{}/DATA/THERMO/{}_{}_E_ST.npy'.format(directory, model.lower(), nframe), 'r') as infile:
				ENERGY[n], ENERGY_ERR[n], _, _, _, _, TEMP[n], TEMP_ERR[n], TENSION[n], TENSION_ERR[n] = np.load(infile)
			with file('{}/DATA/THERMO/{}_{}_TOT_E_ST.npy'.format(directory, model.lower(), nframe), 'r') as infile:
        			_, _, _, tot_tension, _ = np.load(infile)

			VAR_TENSION[n] = np.var(tot_tension)#TENSION_ERR[n]**2

		ENERGY *= e_constant
		ENERGY_ERR *= e_constant
		TENSION *= st_constant
		TENSION_ERR *= st_constant
		VAR_TENSION *= st_constant**2

		graphs.plot_graphs_thermo(ENERGY, ENERGY_ERR, TENSION, TENSION_ERR, VAR_TENSION, N_RANGE, A_RANGE, AN_RANGE, Z_RANGE, l_constant, COLOUR[i])

		TOT_ENERGY += list(ENERGY)
		TOT_ENERGY_ERR += list(ENERGY_ERR)
		TOT_TEMP += list(TEMP)
		TOT_TEMP_ERR += list(TEMP_ERR)
		TOT_TENSION += list(TENSION)
		TOT_TENSION_ERR += list(TENSION_ERR)

		TOT_Z_RANGE += list(Z_RANGE)
		TOT_A_RANGE += list(A_RANGE)
		TOT_N_RANGE += list(N_RANGE)
		TOT_AN_RANGE += list(AN_RANGE)
		
		TOT_VAR_TENSION += list(VAR_TENSION)
		TOT_DEN_RANGE += list(DEN_RANGE)
		TOT_OMEGA_RANGE + list(OMEGA_RANGE)

	TOT_A_RANGE = np.array(TOT_A_RANGE)
	TOT_N_RANGE = np.array(TOT_N_RANGE)
	TOT_Z_RANGE = np.array(TOT_Z_RANGE)
	TOT_AN_RANGE = np.array(TOT_AN_RANGE)

	TOT_ZA_RANGE = np.sqrt(TOT_Z_RANGE / TOT_A_RANGE)

	TOT_DEN_RANGE = np.array(TOT_DEN_RANGE)

	c_energy = thermo.get_U0(model, T, rc, csize, sigma, epsilon)
	c_energy *= e_constant

	print np.mean(TOT_DEN_RANGE)

	if model.upper() == 'ARGON' and red_units == True:
		TOT_DEN_RANGE *= 1. / np.sum(M) * con.N_A * 1E-24 / l_constant**3	
	else: TOT_DEN_RANGE *= 1. / np.sum(M) * con.N_A * 1E6

	print np.mean(TOT_DEN_RANGE)

	NI_ARRAY = np.array([thermo.NI_func(TOT_Z_RANGE[x], TOT_A_RANGE[x], TOT_DEN_RANGE[x], float(rc) * l_constant) for x in range(len(TOT_Z_RANGE))])

	print NI_ARRAY

	VAR_X = np.array(TOT_VAR_TENSION) * np.array(TOT_A_RANGE)**2 / NI_ARRAY

	print VAR_X / e_constant**2

	z_range = np.linspace(np.min(TOT_Z_RANGE), np.max(TOT_Z_RANGE), len(TOT_Z_RANGE))
	a_range = np.linspace(np.min(TOT_A_RANGE), np.max(TOT_A_RANGE), len(TOT_A_RANGE))
	y_data_za = np.sqrt(np.array([thermo.NI_func(TOT_Z_RANGE[x], TOT_A_RANGE[x], np.mean(TOT_DEN_RANGE), float(rc)*l_constant) * np.mean(VAR_X) / TOT_A_RANGE[x]**2 for x in range(len(TOT_ZA_RANGE))]))

	m, c, r_value, p_value, std_err = stats.linregress(TOT_ZA_RANGE , np.array(TOT_TENSION_ERR))
	y_data = [m * x + c for x in TOT_ZA_RANGE]

	TOT_ENERGY = np.array(TOT_ENERGY)
	TOT_ENERGY_ERR = np.array(TOT_ENERGY_ERR)
	TOT_TENSION = np.array(TOT_TENSION)
	TOT_TENSION_ERR = np.array(TOT_TENSION_ERR)

	gamma_err = np.sum(1. / TOT_TENSION_ERR**2)
	av_gamma = np.sum(TOT_TENSION / (TOT_TENSION_ERR**2 * gamma_err))

	print "Surface TENSION {} = {} ({})".format(model.upper(), av_gamma, np.sqrt(np.mean(np.array(TOT_TENSION_ERR)**2)))
	if model.upper() == 'ARGON' and red_units == True:
		print "Average var_X = {}".format(np.mean(VAR_X))
		print "Average density = {}".format(np.mean(TOT_DEN_RANGE))
	else: 
		print "Average var_X = {} kJ^2 mol^-2".format(np.mean(VAR_X) * 1E-12 * con.N_A**2)#conv_mJ_kcalmol**2)
		print "Average std_X = {} kJ mol^-1".format(np.sqrt(np.mean(VAR_X)) * 1E-6 * con.N_A)#conv_mJ_kcalmol**2)
		print "Average density = {} g cm^-3".format(np.mean(TOT_DEN_RANGE) * np.sum(M) / con.N_A * 1E-6)


	an_range = np.linspace(np.min(TOT_AN_RANGE), np.max(TOT_AN_RANGE), len(TOT_AN_RANGE))
	m, c, r_value, p_value, std_err = stats.linregress(TOT_AN_RANGE , TOT_ENERGY)
	ydata = map(lambda x: x * m + c, an_range)

	popt, pcov = curve_fit(ut.linear, TOT_AN_RANGE, TOT_ENERGY, sigma = TOT_ENERGY_ERR)
	print popt, pcov
	error_Us = np.sqrt(pcov[0][0])
	if model.upper() != 'ARGON' or red_units == False:
		m = popt[0] * 1E6
		error_Us = error_Us * 1E6
	print error_Us, std_err, r_value, np.mean(TOT_ENERGY), np.sqrt(np.mean(TOT_ENERGY_ERR**2))
	error_Ss = np.sqrt(1. / np.mean(TOT_TEMP)**2 * (error_Us**2 + np.mean(TOT_TENSION_ERR**2) + (m - av_gamma)**2 / np.mean(TOT_TEMP)**2 * np.mean(np.array(TOT_TEMP_ERR)**2)))
	ydata = map(lambda x: x * popt[0] + popt[1], an_range)

	print "\nUsing (U/N) vs (A/N):"
	print "Surface ENERGY {} = {} ({})".format(model.upper(), m , error_Us)
	print "Surface ENTROPY {} = {} ({})".format(model.upper(), (m - av_gamma) / np.mean(TOT_TEMP), error_Ss )
	print "INTERCEPT: {} ({})  CUBIC ENERGY: {} ".format(c, np.sqrt(pcov[1][1]), c_energy)

	graphs.print_average_graphs_thermo(model, rc, csize, red_units, TOT_ZA_RANGE, y_data_za, an_range, ydata)

	plt.show()


elif func.upper() == 'E':

	import sys
	import mdtraj as md
	import density as den
	import orientational as ori
	import dielectric as die
	import ellipsometry as ellips
	import intrinsic_sampling_method as ism

	import graphs
	from scipy import constants as con
	import matplotlib.pyplot as plt

	print "\n-------------ELLIPSOMETRY PREDICTION ROUINE-----------\n"

	TYPE = 'SLAB'

	if model.upper() == 'AMOEBA': 
		csize = 50
		root = '/data/fl7g13/OpenMM/WATER/{}/T_{}_K/CUT_{}_A/{}'.format(model.upper(), T, rc, TYPE.upper())
	else:
		if model.upper() == 'DMSO': csize = 120
		elif model.upper() == 'ETHANOL': csize = 100
		elif model.upper() == 'METHANOL': csize = 100
		else: csize = 80
		
		root = '{}/{}'.format(root, TYPE.upper())

	epsilon = np.max(LJ[0]) * 4.184
	sigma = np.max(LJ[1])

	if model.upper() not in ['SPCE', 'TIP3P', 'TIP4P2005', 'AMOEBA']:
		directory = '{}/CUBE'.format(root)
		if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))

		rad_dist = bool(raw_input("PERFORM RADIAL DISTRIBUTION? (Y/N): ").upper() == 'Y')

		if rad_dist:
			print "\n-------------CUBIC RADIAL DISTRIBUTION-----------\n"
	
			if not os.path.exists("{}/DATA/DEN".format(directory)): os.mkdir("{}/DATA/DEN".format(directory))

			traj = ut.load_nc(root, 'CUBE', model, csize, 'cube')

			natom = int(traj.n_atoms)
			nmol = int(traj.n_residues)
			ntraj = int(traj.n_frames)
			DIM = np.array(traj.unitcell_lengths[0]) * 10

			nimage = 10#ntraj

			lslice = 0.01
			max_r = np.min(DIM) / 2.
			nslice = int(max_r / lslice)

			ow_all = False
			ow_count = False

			if os.path.exists('{}/DATA/DEN/{}_{}_{}_{}_RDEN.txt'.format(directory, model.lower(), csize, nslice, nimage)) and not ow_all:
				print "FILE FOUND '{}/DATA/DEN/{}_{}_{}_{}_RDEN.txt".format(directory, model.lower(), csize, nslice, nimage)
				if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):
					ow_count = bool(raw_input("OVERWRITE COUNT? (Y/N): ").upper() == 'Y')
					den.radial_dist(traj, directory, model, nimage, max_r, lslice, nslice, natom, nmol, nsite, AT, M, csize, DIM, com, ow_count)	
			else: den.radial_dist(traj, directory, model, nimage, max_r, lslice, nslice, natom, nmol, nsite, AT, M, csize, DIM, com, ow_all)

			with open('{}/DATA/DEN/{}_{}_{}_{}_RDEN.txt'.format(directory, model.lower(), csize, nslice, nimage), 'r') as infile:
				av_density_array = np.loadtxt(infile)

			mol_sigma = 2**(1./6) * av_density_array[0].argmax() * lslice

			plt.plot(np.linspace(0, max_r ,nslice), av_density_array[0])
			plt.show()

			print "molecular sigma = {}".format(mol_sigma)
	
	directory = '{}/{}'.format(root, folder.upper())
	data_dir = '{}/DATA'.format(directory)

	if not os.path.exists(data_dir): os.mkdir(data_dir)

	print "\n----------BUILDING SURFACE POSITIONAL ARRAYS-----------\n"

	ow_pos = bool(raw_input("OVERWRITE AT MOL POSITIONS? (Y/N): ").upper() == 'Y')

	natom, nmol, ntraj, DIM, COM = ut.get_sim_param(root, directory, data_dir, model, nsite, suffix, csize, M, com, ow_pos, False)

	lslice = 0.05 * sigma
	nslice = int(DIM[2] / lslice)
	vlim = 3
	ncube = 3
	phi = 5E-8
	psi = phi * DIM[0] * DIM[1] 

	mol_sigma, ns = ut.get_ism_constants(model, sigma)
	npi = 50

	if model.upper() == 'METHANOL': nframe = 3500
	elif model.upper() == 'TIP4P2005': nframe = 3500
	elif model.upper() == 'ARGON': nframe = 100
	else: nframe = ntraj

	print "natom = {}, nmol = {}, nframe = {}\nDIM = {}, lslice = {}, nslice = {}\nmol mass = {}\n".format(natom, nmol, nframe, DIM, lslice, nslice, np.sum(M))

	print "\n-----------STARTING DENSITY PROFILE------------\n"
	
	if not os.path.exists("{}/DEN".format(data_dir)): os.mkdir("{}/DEN".format(data_dir))

	ow_all = False
	ow_count = False

	file_name_den = '{}_{}_{}'.format(model.lower(), nslice, nframe)
	if os.path.exists('{}/DEN/{}_DEN.npy'.format(data_dir, file_name_den)) and not ow_all:
		print "FILE FOUND '{}/DEN/{}_DEN.npy".format(data_dir, file_name_den)
		if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):
			ow_count = bool(raw_input("OVERWRITE COUNT? (Y/N): ").upper() == 'Y')
			den.density_profile(data_dir, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, COM, DIM, nslice, ow_count)	
	else: den.density_profile(data_dir, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, COM, DIM, nslice, ow_all)
	#graphs.print_graphs_density(directory, model, nsite, AT, M, nslice, rc, csize, folder, suffix, nframe, DIM)

	ow_all = False
	ow_angle = False
	ow_polar = False

	if model.upper() != 'ARGON':

		print "\n--------STARTING ORIENTATIONAL PROFILE--------\n"

		if not os.path.exists("{}/EULER".format(data_dir)): os.mkdir("{}/EULER".format(data_dir))

		file_name_euler = '{}_{}_{}_{}'.format(model.lower(), a_type, nslice, nframe)
		if os.path.exists('{}/EULER/{}_EUL.npy'.format(data_dir, file_name_euler)) and not ow_all:
			print 'FILE FOUND {}/EULER/{}_EUL.npy'.format(data_dir, file_name_euler)
			if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):  
				ow_angle = bool(raw_input("OVERWRITE ANGLES? (Y/N): ").upper() == 'Y')
				ow_polar = bool(raw_input("OVERWRITE POLARISABILITY? (Y/N): ").upper() == 'Y') 
				ori.euler_profile(data_dir, nframe, nslice, nmol, model, csize, suffix, AT, Q, M, LJ, COM, DIM, nsite, a_type, npi, ow_angle, ow_polar)
		else: ori.euler_profile(data_dir, nframe, nslice, nmol, model, csize, suffix, AT, Q, M, LJ, COM, DIM, nsite, a_type, npi, ow_all, ow_all)
		#graphs.print_graphs_orientational(directory, model, nsite, AT, nslice, a_type, rc, csize, folder, suffix, nframe, DIM)

	ow_all = False

	print "\n-------STARTING DIELECTRIC PROFILE--------\n"

	if not os.path.exists("{}/DIELEC".format(data_dir)): os.mkdir("{}/DIELEC".format(data_dir))
	if not os.path.exists("{}/ELLIP".format(data_dir)): os.mkdir("{}/ELLIP".format(data_dir))

	file_name_die = '{}_{}_{}_{}'.format(model.lower(), a_type, nslice, nframe)
	if os.path.exists('{}/DIELEC/{}_DIE.npy'.format(data_dir, file_name_die)) and not ow_all:
		print 'FILE FOUND {}/DIELEC/{}_DIE.npy'.format(data_dir, file_name_die)
		if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):   
			#ow_ecount = bool(raw_input("OVERWRITE ECOUNT? (Y/N): ").upper() == 'Y')
			#ow_acount = bool(raw_input("OVERWRITE ACOUNT? (Y/N): ").upper() == 'Y') 
			die.dielectric_refractive_index(data_dir, model, csize, AT, sigma, mol_sigma, nslice, nframe, a_type, DIM)
	else: die.dielectric_refractive_index(data_dir, model, csize, AT, sigma, mol_sigma, nslice, nframe, a_type, DIM)
	#graphs.print_graphs_dielectric(directory, model, nsite, AT, nslice, a_type, rc, csize, folder, suffix, nframe, DIM)
	
	print "\n------STARTING INTRINSIC SAMPLING METHODS-------\n"

	qu = 2 * np.pi / mol_sigma
	ql = 2 * np.pi / np.sqrt(DIM[0] * DIM[1])
	nm = int(qu / ql)

	if bool(raw_input("PERFORM NS OPTIMISATION? (Y/N): ").upper() == 'Y'): ns = ism.optimise_ns(directory, model.lower(), csize, nmol, nsite, nm, phi, vlim, ncube, DIM, COM, M, mol_sigma, ns-0.20, ns + 0.25)

        #n0 = int(ns * mol_sigma**2 + 0.5)
        #Np = int(DIM[0] * DIM[1] * n0)
	print DIM[0] / mol_sigma, DIM[1]/mol_sigma
	n0 = int(DIM[0] * DIM[1] * ns / mol_sigma**2)

	ow_dist = False
	ow_count = False
	ow_angle = False
	ow_polar = False

	QM = range(1, nm+1)
	print QM
	print "{:12s} | {:12s} | {:12s} | {:12s} | {:12s} | {:12s}".format('nm', 'ns', 'n0', 'phi', "lambda", "lambda (nm)")
	print "-" * 14 * 5 
	for qm in QM: print "{:12d} | {:12.4f} | {:12d} | {:12.8f} | {:12.4f} | {:12.4f}".format(qm, ns, n0, phi, qu / (qm*ql), mol_sigma * qu / (10*qm*ql))
	print ""

	if not os.path.exists("{}/INTDEN".format(data_dir)): os.mkdir("{}/INTDEN".format(data_dir))
	if not os.path.exists("{}/INTTHERMO".format(data_dir)): os.mkdir("{}/INTTHERMO".format(data_dir))
	if not os.path.exists("{}/INTPOS".format(data_dir)): os.mkdir("{}/INTPOS".format(data_dir))
	if not os.path.exists("{}/INTEULER".format(data_dir)): os.mkdir("{}/INTEULER".format(data_dir))
	if not os.path.exists("{}/INTDIELEC".format(data_dir)): os.mkdir("{}/INTDIELEC".format(data_dir))

	#ism.intrinsic_optimisation(directory, model, csize, rc, suffix, nframe, nslice, nmol, nsite, AT, M, T, sigma, epsilon, mol_sigma, ncube, DIM, COM, nm, n0, vlim, ow_coeff, ow_count)

	ow_coeff = bool(raw_input("OVERWRITE ACOEFF? (Y/N): ").upper() == 'Y')
	ow_recon = bool(raw_input("OVERWRITE RECON ACOEFF? (Y/N): ").upper() == 'Y')
	ow_pos = bool(raw_input("OVERWRITE POSITIONS? (Y/N): ").upper() == 'Y')
	ow_dxdyz = bool(raw_input("OVERWRITE DERIVATIVES? (Y/N): ").upper() == 'Y')
	ow_profile = bool(raw_input("OVERWRITE INTRINSIC PROFILES? (Y/N): ").upper() == 'Y')
	if ow_profile:
		ow_dist = bool(raw_input("OVERWRITE INTRINSIC DISTRIBUTIONS? (Y/N): ").upper() == 'Y')
		ow_count = bool(raw_input("OVERWRITE INTRINSIC COUNT? (Y/N): ").upper() == 'Y')
		if model.upper() != 'ARGON':
			ow_angle = bool(raw_input("OVERWRITE INTRINSIC ANGLES? (Y/N): ").upper() == 'Y')
			ow_polar = bool(raw_input("OVERWRITE INTRINSIC POLARISABILITY? (Y/N): ").upper() == 'Y') 	
	pos_1, pos_2 = ism.intrinsic_profile(data_dir, model, csize, nframe, natom, nmol, nsite, AT, M, a_type, mol_sigma, COM, DIM, nslice, ncube, nm, QM, n0, phi, npi, vlim, ow_profile, ow_coeff, ow_recon, ow_pos, ow_dxdyz, ow_dist, ow_count, ow_angle, ow_polar)

	graphs.print_graphs_intrinsic_density(data_dir, model, nsite, AT, M, nslice, nm, QM, n0, phi, rc, csize, folder, suffix, nframe, DIM, pos_1, pos_2)
	if model.upper() != 'ARGON': graphs.print_graphs_intrinsic_orientational(directory, model, nsite, AT, nslice, nm, QM, n0, phi, a_type, rc, csize, folder, suffix, nframe, DIM, pos_1, pos_2)
	graphs.print_graphs_intrinsic_dielectric(directory, model, nsite, AT, nslice, nm, QM, n0, phi, a_type, rc, csize, folder, suffix, nframe, DIM, pos_1, pos_2)

	print "\n-------STARTING ELLIPSOMETRY PREDICTIONS--------\n"

	ellips.transfer_matrix(data_dir, model, csize, AT, sigma, mol_sigma, nslice, nframe, a_type, nm, QM, n0, phi, DIM, rc)

