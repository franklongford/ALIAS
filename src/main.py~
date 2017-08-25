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
cutoff = int(raw_input("Cutoff: (A) "))
func = raw_input("Function:\nTest or Slab? (T, S): ")

if model.upper() in['ARGON', 'METHANOL', 'ETHANOL', 'DMSO']: root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A'.format(model.upper(), T, cutoff)
else: root = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A'.format(model.upper(), T, cutoff)

if func.upper() == 'T':
	TYPE = raw_input("Width (W), Area (A) or Cubic (C) variation: ")
	force = raw_input("VDW Force corrections? (Y/N): ")

	root = '{}/{}_TEST'.format(root, TYPE.upper())

	if force.upper() == 'Y': folder = 'SURFACE_2'
	else: folder = 'SURFACE' 

	suffix = 'surface'
	csize = 50

	if model.upper() == 'ARGON':
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

	if model.upper() == 'TIP4P2005':
		if TYPE.upper() == 'W': 
			nfolder = 29
			sfolder = 0 
		if TYPE.upper() == 'A': nfolder = 25
		if TYPE.upper() == 'C':
		        nfolder = 2
		        csize = 35
		if T != 298: nfolder = 1	

	if model.upper() in ['SPCE', 'TIP3P']:
		if TYPE.upper() == 'W':
			nfolder = 1#29
			sfolder = 0

	print ""
	build = False#bool(raw_input("Make input files or Analyse?").upper() == 'Y')

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
						_, _, ntraj, _, _, dim_Z = np.loadtxt(infile)

					ntraj = int(ntraj)
					nslice = int(dim_Z / lslice)

					if os.path.exists('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, ntraj)) and not ow_all:
						print "FILE FOUND '{}/DATA/DEN/{}_{}_{}_{}_DEN.txt".format(directory, model.lower(), csize, nslice, ntraj)
						ow_den = bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y')
						if ow_den: ow_count = bool(raw_input("OVERWRITE COUNT? (Y/N): ").upper() == 'Y')

				if ow_den or ow_all:
					import density as den

					print '{}/{}/{}_{}_{}.nc'.format(root_, folder.upper(), model.lower(), csize, suffix)
					traj = ut.load_nc(root_, folder, model, csize, suffix)

					if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))

					natom = int(traj.n_atoms)
					nmol = int(traj.n_residues)
					ntraj = int(traj.n_frames)
					DIM = np.array(traj.unitcell_lengths[0]) * 10

					with file('{}/DATA/parameters.txt'.format(directory), 'w') as outfile:
						np.savetxt(outfile, [natom, nmol, ntraj, DIM[0], DIM[1], DIM[2]])

					nslice = int(DIM[2] / lslice)

					if not os.path.exists("{}/DATA/DEN".format(directory)): os.mkdir("{}/DATA/DEN".format(directory))
						
					if ow_all: den.density_thermo(traj, directory, model, csize, suffix, ntraj, natom, nmol, nsite, AT, M, com, DIM, nslice, ow_all)
					else: den.density_thermo(traj, directory, model, csize, suffix, ntraj, natom, nmol, nsite, AT, M, com, DIM, nslice, ow_count)

				print ""

		if TASK.upper() == 'IS':

			import density as den
			import thermodynamics as thermo
			import intrinsic_sampling_method as ism

			rc = float(cutoff)
			"Conversion of length and surface tension units"

			epsilon = LJ[0] * 4.184
			sigma = np.max(LJ[1])

			e_constant, st_constant, l_constant = ut.get_thermo_constants(model, LJ)

			ow_all =  bool(raw_input("OVERWRITE ALL ACOEFF? (Y/N): ").upper() == 'Y')
			ow_pos = bool(raw_input("OVERWRITE AT MOL POSITIONS? (Y/N): ").upper() == 'Y')
			
			lslice = 0.05 * sigma

			E_N = np.zeros(nfolder)
			POT_N = np.zeros(nfolder)
			KIN_N = np.zeros(nfolder)
			A_N = np.zeros(nfolder)
			ST = np.zeros(nfolder)
			TEMP = np.zeros(nfolder)

			intA = [[] for n in range(nfolder-sfolder)]
			intA_N = [[] for n in range(nfolder-sfolder)]
			gamma_lv = [[] for n in range(nfolder-sfolder)]
			gamma_q = [[] for n in range(nfolder-sfolder)]
			av_stdh = [[] for n in range(nfolder-sfolder)]
			tot_q_set = []
			mol_den = [[] for n in range(nfolder-sfolder)]
			cw_mol_den = [[] for n in range(nfolder-sfolder)]

			COLOUR = ['b', 'r', 'g', 'cyan', 'orange']
			
			for i, n in enumerate(range(sfolder, nfolder)):
				root_ = '{}/{}_{}'.format(root, TYPE.upper(), n)
				directory = '{}/{}'.format(root_, folder.upper())

				if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))
				if not os.path.exists("{}/DATA/INTDEN".format(directory)): os.mkdir("{}/DATA/INTDEN".format(directory))
				if not os.path.exists("{}/DATA/INTPOS".format(directory)): os.mkdir("{}/DATA/INTPOS".format(directory))

				natom, nmol, ntraj, DIM, COM = ut.get_sim_param(root_, directory, model, nsite, suffix, csize, M, com, False)

				if model.upper() in ['SPCE', 'TIP3P', 'TIP4P2005', 'ARGON', 'AMOEBA']: mol_sigma = sigma
				elif model.upper() == 'METHANOL': mol_sigma = 3.83
				elif model.upper() == 'ETHANOL': mol_sigma = 4.57
				elif model.upper() == 'DMSO': mol_sigma = 5.72

				lslice = 0.05 * sigma
				nslice = int(DIM[2] / lslice)
				vlim = 3
				ncube = 3
				max_r = 1.5 * mol_sigma
				tau = 0.4 * mol_sigma


				if not os.path.exists('{}/DATA/DEN/{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), csize, nslice, 4000)):
					den.density_profile(directory, model, csize, suffix, 4000, natom, nmol, nsite, AT, M, COM, DIM, nslice, False)
				if not os.path.exists('{}/DATA/ENERGY_TENSION/{}_{}_EST.txt'.format(directory, model.lower(), csize)):
					thermo.get_thermo(directory, model, csize, suffix, nslice, 4000, DIM, nmol, rc, sigma, epsilon, l_constant, False)

				with file('{}/DATA/ENERGY_TENSION/{}_{}_EST.txt'.format(directory, model.lower(), csize), 'r') as infile:
					E_N[i], _, POT_N[i], _, KIN_N[i], _, TEMP[i], _, ST[i], _ = np.loadtxt(infile)
				with file('{}/DATA/ENERGY_TENSION/{}_{}_TOTEST.txt'.format(directory, model.lower(), csize), 'r') as infile:
                			TOTAL_ENERGY, TOTAL_POTENTIAL, TOTAL_KINETIC, TOTAL_TENSION, TOTAL_TEMP = np.loadtxt(infile)			
	
				A_N[i] = 2 * DIM[0]*DIM[1] * l_constant**2 / (nmol / con.N_A)
			
				print DIM

				if model.upper() == 'ARGON': 
					ns = 0.8
				elif model.upper() == 'SPCE': 
					ns = 1.20
				elif model.upper() == 'TIP4P2005': 
					ns = 1.15

				phi = 5E-8
				nframe = 50

				print "{:12s} {:12s} {:12s} {:12s} {:12s} {:12s} {:12s} {:12s} {:12s}".format('nm', 'qm', 'intA 1', 'intA 2', "std h 1", "std h 2", "gamma", "gamma_lv", "kappa")

				qu = 2 * np.pi / mol_sigma
				ql = 2 * np.pi / np.sqrt(DIM[0] * DIM[1]) 
				nm = int(qu / ql)

				#ns = ism.optimise_ns(directory, model.lower(), csize, nmol, nsite, nm, phi, vlim, ncube, DIM, COM, M, mol_sigma, ns-0.25, ns + 0.25)
				n0 = int(DIM[0] * DIM[1] * ns / mol_sigma**2)

				auv1_2 = np.zeros((2*nm+1)**2)
				auv2_2 = np.zeros((2*nm+1)**2)

				mean_auv1 = np.zeros(nframe)
				mean_auv2 = np.zeros(nframe)

				#ism.intrinsic_profile(directory, model, csize, nframe, natom, nmol, nsite, AT, M, mol_sigma, COM, DIM, nslice, ncube, nm, n0, phi, vlim, False, False, True)

				for frame in range(nframe):
					#auv1, auv2, piv_n1, piv_n2 = ism.intrinsic_surface_test(directory, model, csize, nsite, nmol, ncube, DIM, COM, nm, n0, phi, vlim, mol_sigma, frame, nframe)
					auv1, auv2, piv_n1, piv_n2 = ism.intrinsic_surface(directory, model, csize, nsite, nmol, ncube, DIM, COM, nm, n0, phi, vlim, mol_sigma, M, frame, nframe, ow_all)

					auv1_2 += auv1**2 / nframe
					auv2_2 += auv2**2 / nframe

					mean_auv1[frame] = auv1[len(auv1)/2]
					mean_auv2[frame] = auv2[len(auv2)/2]

				QM = [0] + range(1, nm/2) + [nm]

				for j, qm in enumerate(QM):

					q_set = []
					q2_set = []

					for u in xrange(-qm, qm):
						for v in xrange(-qm, qm):
							q = 4 * np.pi**2 * (u**2 / DIM[0]**2 + v**2/DIM[1]**2)
							q2 = u**2 * DIM[1]/DIM[0] + v**2 * DIM[0]/DIM[1]

							if q2 not in q2_set:
								q_set.append(q)
								q2_set.append(np.round(q2, 4))

					q_set = np.sqrt(np.sort(q_set, axis=None))
					q2_set = np.sort(q2_set, axis=None)
					gamma_q[i].append(np.zeros(len(q_set)))

					tot_q_set.append(q_set)

					l_cut_q = 0.1
					u_cut_q = 2 / mol_sigma
					l_cut_index = 1
					u_cut_index = 1
					for k, q in enumerate(q_set):
						if q < l_cut_q:
							l_cut_index = k
						if q < u_cut_q:
							u_cut_index = k

					#ism.intrinsic_profile(directory, model, csize, nframe, natom, nmol, nsite, AT, M, mol_sigma, COM, DIM, nslice, ncube, nm, n0, phi, vlim, False, False, True)
					
					density = np.zeros(nslice)

					for frame in range(nframe):
						count_array = ism.intrinsic_density(directory, COM, model, csize, nm, qm, n0, phi, frame, nslice, nsite, AT, DIM, M, False, False)
						density += count_array[-2] / (lslice * DIM[0] * DIM[1] * nframe)

					mol_den[i].append(density)

					Delta1 = (ut.sum_auv_2(auv1_2, nm, qm) - np.mean(mean_auv1)**2)
					cw_arrays = ut.gaussian_smoothing([density, np.zeros(nslice)], [np.mean(mean_auv1), 0], [Delta1, 0], DIM, nslice)

					cw_mol_den[i].append(cw_arrays[0])

					intA_1_2 = ism.slice_area(auv1_2, auv2_2, nm, qm, DIM)
					intA[i].append(np.mean(intA_1_2))

					gamma_list, gamma_hist = ism.gamma_q_auv((auv1_2 + auv2_2) / 2., nm, qm, DIM, T, q2_set)

					if qm > 0: opt, cov = curve_fit(lambda x, a, b: a + b * x**2, q_set[l_cut_index:u_cut_index], gamma_hist[l_cut_index:u_cut_index], [1, 1])
					else: opt = [0, 0]
					gamma_lv[i].append(opt[0])
					gamma_q[i][j] += gamma_hist

					Delta1 = (ut.sum_auv_2(auv1_2, nm, qm) - np.mean(mean_auv1)**2)
					Delta2 = (ut.sum_auv_2(auv2_2, nm, qm) - np.mean(mean_auv2)**2)

					av_stdh[i].append((np.sqrt(Delta1) + np.sqrt(Delta2)) / 2.)

					#plt.figure(i)
					#plt.scatter(q_set, gamma_hist, c=COLOUR[j%len(COLOUR)])
					#plt.plot(q_set, [opt[0] + opt[1] * x**2 for x in q_set], c=COLOUR[j%len(COLOUR)])
					#plt.show()

					intA_N[i].append(intA[i][j] * A_N[i])
					
					print "{:12d} {:12d} {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:12.4f}".format(nm, qm, intA_1_2[0], np.sqrt(Delta1), np.sqrt(Delta2), ST[i], opt[0], opt[1])


			gamma_q = np.array(gamma_q)
			mol_den = np.array(mol_den)
			cw_mol_den = np.array(cw_mol_den)
			for n in xrange(nfolder-1):
				gamma_q[0] += gamma_q[n+1] / nfolder
				mol_den[0] += mol_den[n+1] / nfolder
				cw_mol_den[0] += cw_mol_den[n+1] / nfolder
			gamma_q = gamma_q[0]
			mol_den = mol_den[0]
			cw_mol_den = cw_mol_den[0]

			av_gamma = np.mean(ST)
			av_temp = np.mean(TEMP)

			m_e, c_e, r_value, p_value, std_err = stats.linregress(A_N, E_N)

			plt.figure(nfolder)
			plt.plot(A_N, [c_e + m_e * x for x in A_N], c='black', linestyle='dashed')
			plt.scatter(A_N, E_N, c='black', marker='x')

			m_pot, c_pot, r_value, p_value, std_err = stats.linregress(A_N, POT_N)

			plt.figure(nfolder + 1)
			plt.plot(A_N, [c_pot + m_pot * x for x in A_N], c='blue', linestyle='dashed')
			plt.scatter(A_N, POT_N, c='blue', marker='x')

			gamma_c = []
			kappa_c = []

			print "\nSURFACE THERMODYNAMICS"
			print " {:12s} | {:12s} | {:12s} | {:12s} | {:15s} | {:15s} | {:15s} | {:15s}" .format('qm', 'av intA', 'std h', 'S h', 'gamma 1 (mJ m-2)', 'kappa (mJ)', 'gamma 2 (mJ m-2)', 'gamma 3 (mJ m-2)')
			print "-" * 7 * 19
			print " {:12s} | {:12s} | {:12s} | {:12s} | {:15.3f} | {:15s}  {:15s} | {:15s}".format('-', '-', '-', '-', av_gamma, '-', '-', '-' )
			print "-" * 7 * 19
	
			intA = np.transpose(intA)
			intA_N = np.transpose(intA_N)
			av_stdh = np.transpose(av_stdh)

			Z = np.linspace(-DIM[2]/2, DIM[2]/2, nslice)

			for i, qm in enumerate(QM):

				plt.figure(nfolder + 2)
				m_pot, c_pot, r_value, p_value, std_err = stats.linregress(intA_N[i], POT_N)
				plt.plot(intA_N[i], [c_pot + m_pot * x for x in intA_N[i]], c=COLOUR[i%len(COLOUR)], linestyle='dashed') 
				plt.scatter(intA_N[i], POT_N, c=COLOUR[i%len(COLOUR)], marker='x')

				plt.figure(nfolder + 3)
				m_kin, c_kin, r_value, p_value, std_err = stats.linregress(intA_N[i], KIN_N)
				plt.plot(intA_N[i], [c_kin + m_kin * x for x in intA_N[i]], c=COLOUR[i%len(COLOUR)], linestyle='dashed') 
				plt.scatter(intA_N[i], KIN_N, c=COLOUR[i%len(COLOUR)], marker='x')

				plt.figure(nfolder + 4)
				if qm > 0: opt, cov = curve_fit(lambda x, a, b: a + b * x**2, tot_q_set[i][l_cut_index:u_cut_index], gamma_q[i][l_cut_index:u_cut_index], [1, 1])
				else: opt = [0, 0]
				plt.scatter(tot_q_set[i][l_cut_index:], gamma_q[i][l_cut_index:], c=COLOUR[i%len(COLOUR)])
				plt.plot(tot_q_set[i], [opt[0] + opt[1] * x**2 for x in tot_q_set[i]], c=COLOUR[i%len(COLOUR)])
				#plt.axis([0, tot_q_set[i][-1], -250, 250])
				#plt.scatter(np.sqrt(q2_set)[1:], gamma_q[i][1:], c=COLOUR[i%len(COLOUR)])
                                #plt.plot(np.sqrt(q2_set)[1:], [opt[0] + opt[1] * x**2 for x in q_set[1:]], c=COLOUR[i%len(COLOUR)])
                                #plt.axis([0, np.sqrt(q2_set)[-1], 0, 400])
				plt.savefig('/home/fl7g13/Documents/Thesis/Figures/test_gamma_q_{}_{}.png'.format(model.lower(), nfolder))
	
				plt.figure(nfolder+5)
				plt.plot(Z, mol_den[i])

				plt.figure(nfolder+6)
				plt.plot(Z, cw_mol_den[i])

				#plt.figure(nfolder + 1)
				#m, c, r_value, p_value, std_err = stats.linregress(intA_N[i], E_N)
				#plt.scatter(intA_N[i], E_N, c=COLOUR[i%len(COLOUR)])

				print " {:12d} | {:12.4f} | {:12.4f} | {:12.4f} | {:15.3f} | {:15.3f} | {:15.3f} | {:15.3f}".format(qm, np.mean(intA[i]), np.mean(av_stdh[i]), 
															   np.log(np.sqrt(np.mean(av_stdh[i]**2) * np.pi * 2 * np.exp(1))), 
															   opt[0], opt[1], m_pot * 1E6,  -m_kin * 1E6,)
				gamma_c.append(opt[0])
				kappa_c.append(opt[1])			

			plt.figure(nfolder + 7)
			plt.plot(QM, gamma_c, c='black', linestyle='dashed')
			plt.scatter(QM, gamma_c, c='black', marker='x')
			plt.plot(QM, kappa_c, c='red', linestyle='dashed')
			plt.scatter(QM, kappa_c, c='red', marker='x')

			plt.show()


		elif TASK.upper() == 'T':

			import thermodynamics as thermo

			rc = float(cutoff)

			"Conversion of length and surface tension units"
			if model.upper() == 'ARGON':
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


elif func.upper() == 'S':

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

	TYPE = 'SLAB'

	if model.upper() == 'AMOEBA': 
		csize = 50
		root = '/data/fl7g13/OpenMM/WATER/{}/T_{}_K/CUT_{}_A/{}'.format(model.upper(), T, cutoff, TYPE.upper())
	else:
		if model.upper() == 'DMSO': csize = 120
		elif model.upper() == 'ETHANOL': csize = 100
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
	if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))

	print "\n----------BUILDING SURFACE POSITIONAL ARRAYS-----------\n"

	ow_pos = bool(raw_input("OVERWRITE AT MOL POSITIONS? (Y/N): ").upper() == 'Y')

	natom, nmol, ntraj, DIM, COM = ut.get_sim_param(root, directory, model, nsite, suffix, csize, M, com, False)

	lslice = 0.05 * sigma
	nslice = int(DIM[2] / lslice)
	vlim = 3
	ncube = 3

	mol_sigma, ns, phi = ut.get_ism_constants(model, sigma)
	npi = 50

	nframe = 200#ntraj

	print "\n-----------STARTING DENSITY PROFILE------------\n"
	
	if not os.path.exists("{}/DATA/DEN".format(directory)): os.mkdir("{}/DATA/DEN".format(directory))

	ow_all = False
	ow_count = False

	if os.path.exists('{}/DATA/DEN/{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nframe)) and not ow_all:
		print "FILE FOUND '{}/DATA/DEN/{}_{}_{}_DEN.txt".format(directory, model.lower(), nslice, nframe)
		if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):
			ow_count = bool(raw_input("OVERWRITE COUNT? (Y/N): ").upper() == 'Y')
			den.density_profile(directory, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, COM, DIM, nslice, ow_count)	
	else: den.density_profile(directory, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, COM, DIM, nslice, ow_all)
	#graphs.print_graphs_density(directory, model, nsite, AT, M, nslice, cutoff, csize, folder, suffix, nframe, DIM)

	ow_all = False
	ow_angle = False
	ow_polar = False

	if model.upper() != 'ARGON':

		print "\n--------STARTING ORIENTATIONAL PROFILE--------\n"

		if not os.path.exists("{}/DATA/EULER".format(directory)): os.mkdir("{}/DATA/EULER".format(directory))

		if os.path.exists('{}/DATA/EULER/{}_{}_{}_{}_EUL.txt'.format(directory, model.lower(), a_type, nslice, nframe)) and not ow_all:
			print 'FILE FOUND {}/DATA/EULER/{}_{}_{}_{}_EUL.txt'.format(directory, model.lower(), a_type, nslice, nframe)
			if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):  
				ow_angle = bool(raw_input("OVERWRITE ANGLES? (Y/N): ").upper() == 'Y')
				ow_polar = bool(raw_input("OVERWRITE POLARISABILITY? (Y/N): ").upper() == 'Y') 
				ori.euler_profile(directory, nframe, nslice, nmol, model, csize, suffix, AT, Q, M, LJ, COM, DIM, nsite, a_type, npi, ow_angle, ow_polar)
		else: ori.euler_profile(directory, nframe, nslice, nmol, model, csize, suffix, AT, Q, M, LJ, COM, DIM, nsite, a_type, npi, ow_all, ow_all)
		#graphs.print_graphs_orientational(directory, model, nsite, AT, nslice, a_type, cutoff, csize, folder, suffix, nframe, DIM)

	ow_all = False

	print "\n-------STARTING DIELECTRIC PROFILE--------\n"

	if not os.path.exists("{}/DATA/DIELEC".format(directory)): os.mkdir("{}/DATA/DIELEC".format(directory))
	if not os.path.exists("{}/DATA/ELLIP".format(directory)): os.mkdir("{}/DATA/ELLIP".format(directory))

	if os.path.exists('{}/DATA/DIELEC/{}_{}_{}_{}_DIE.txt'.format(directory, model.lower(), a_type, nslice, nframe)) and not ow_all:
		print 'FILE FOUND {}/DATA/DIELEC/{}_{}_{}_{}_DIE.txt'.format(directory, model.lower(), a_type, nslice, nframe)
		if bool(raw_input("OVERWRITE? (Y/N): ").upper() == 'Y'):   
			#ow_ecount = bool(raw_input("OVERWRITE ECOUNT? (Y/N): ").upper() == 'Y')
			#ow_acount = bool(raw_input("OVERWRITE ACOUNT? (Y/N): ").upper() == 'Y') 
			die.dielectric_refractive_index(directory, model, csize, AT, sigma, mol_sigma, nslice, nframe, a_type, DIM)
	else: die.dielectric_refractive_index(directory, model, csize, AT, sigma, mol_sigma, nslice, nframe, a_type, DIM)
	#graphs.print_graphs_dielectric(directory, model, nsite, AT, nslice, a_type, cutoff, csize, folder, suffix, nframe, DIM)
	
	print "\n------STARTING INTRINSIC SAMPLING METHODS-------\n"

	qu = 2 * np.pi / mol_sigma
	ql = 2 * np.pi / np.sqrt(DIM[0] * DIM[1])
	nm = int(qu / ql)

	if bool(raw_input("PERFORM NS OPTIMISATION? (Y/N): ").upper() == 'Y'): ns = ism.optimise_ns(directory, model.lower(), csize, nmol, nsite, nm, phi, vlim, ncube, DIM, COM, M, mol_sigma, ns-0.25, ns + 0.25)
	n0 = int(DIM[0] * DIM[1] * ns / mol_sigma**2)

	ow_all = False
	ow_coeff = False
	ow_pos = False
	ow_dxdyz = False
	ow_count = False

	QM = range(1, nm/2+3) +  [nm]

	print QM, ns, n0

	if not os.path.exists("{}/DATA/INTDEN".format(directory)): os.mkdir("{}/DATA/INTDEN".format(directory))
	if not os.path.exists("{}/DATA/INTPOS".format(directory)): os.mkdir("{}/DATA/INTPOS".format(directory))
	if not os.path.exists("{}/DATA/INTEULER".format(directory)): os.mkdir("{}/DATA/INTEULER".format(directory))
	if not os.path.exists("{}/DATA/INTDIELEC".format(directory)): os.mkdir("{}/DATA/INTDIELEC".format(directory))

	#ism.intrinsic_optimisation(directory, model, csize, cutoff, suffix, nframe, nslice, nmol, nsite, AT, M, T, sigma, epsilon, mol_sigma, ncube, DIM, COM, nm, n0, vlim, ow_coeff, ow_count)

	ow_coeff = bool(raw_input("OVERWRITE ACOEFF? (Y/N): ").upper() == 'Y')
	ow_pos = bool(raw_input("OVERWRITE POSITIONS? (Y/N): ").upper() == 'Y')
	ow_dxdyz = bool(raw_input("OVERWRITE DERIVATIVES? (Y/N): ").upper() == 'Y')
	ow_profile = bool(raw_input("OVERWRITE INTRINSIC PROFILES? (Y/N): ").upper() == 'Y')
	if ow_profile:
		ow_count = bool(raw_input("OVERWRITE INTRINSIC COUNT? (Y/N): ").upper() == 'Y')
		if model.upper() != 'ARGON':
			ow_angle = bool(raw_input("OVERWRITE INTRINSIC ANGLES? (Y/N): ").upper() == 'Y')
			ow_polar = bool(raw_input("OVERWRITE INTRINSIC POLARISABILITY? (Y/N): ").upper() == 'Y') 	
	ism.intrinsic_profile(directory, model, csize, nframe, natom, nmol, nsite, AT, M, a_type, mol_sigma, COM, DIM, nslice, ncube, nm, QM, n0, phi, npi, vlim, ow_profile, ow_coeff, ow_pos, ow_dxdyz, ow_count, ow_angle, ow_polar)

	#graphs.print_graphs_intrinsic_density(directory, model, nsite, AT, M, nslice, nm, QM, n0, phi, cutoff, csize, folder, suffix, nframe, DIM)
	#if model.upper() != 'ARGON': graphs.print_graphs_intrinsic_orientational(directory, model, nsite, AT, nslice, nm, QM, n0, phi, a_type, cutoff, csize, folder, suffix, nframe, DIM)
	#graphs.print_graphs_intrinsic_dielectric(directory, model, nsite, AT, nslice, nm, QM, n0, phi, a_type, cutoff, csize, folder, suffix, nframe, DIM)

	print "\n-------STARTING ELLIPSOMETRY PREDICTIONS--------\n"

	ellips.transfer_matrix(directory, model, csize, AT, sigma, nslice, nframe, a_type, nm, QM, n0, phi, DIM, cutoff)

