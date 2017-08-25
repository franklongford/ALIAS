import matplotlib
matplotlib.use('TkAgg')

import Tkinter as Tk
import tkMessageBox
import tkFont
import numpy as np
import os
import utilities as ut
import sys
import subprocess


import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkSimpleDialog


class set_up:

	def __init__(self, pixel):

		
		# Defines the size of the opening window
		# This determines the size ration of everything within the window
		self.pixel = pixel
		self.Lx = self.pixel * 200
		self.Ly = self.pixel * 200
		

		self.load_screen()

	def load_screen(self):
	
		self.master = Tk.Tk()

		#Builds the inputs screen automatically
		input_screen = inputs(self)

		Tk.mainloop()

class inputs():

	def __init__(self, set_up):

		self.set_up = set_up
		self.screen = self.set_up.master
		self.pixel = self.set_up.pixel
		self.Lx = self.set_up.Lx
		self.Ly = self.set_up.Ly
		

		#Creates the title of the input screen
		#PUT IN A FRAME TO CENTRE IT SO ALL LOOKS NICER
		self.title = Tk.Label(self.set_up.master, text = 'ALIAS', font = ('Calibri', 24))
		self.title.grid(row = 0, column = 1)

		#Builds the drop down menu to choose which model you want to test
		self.model_options = ['SPCE', 'ARGON', 'TIP3P', 'TIP4P2005', 'AMOEBA', 'METHANOL', 'ETHANOL', 'DMSO']
		self.model_chosen = Tk.StringVar(self.set_up.master)
		self.model_chosen.set(self.model_options[0]) #Default value

		self.model_menu = apply(Tk.OptionMenu, (self.set_up.master, self.model_chosen) + tuple(self.model_options))
		self.model_menu.pack()
		self.model_menu.grid(column = 0, row = 1)
		

		#Builds a text box to input temperature in kelvin
		self.temp_label = Tk.Label(self.set_up.master, text = 'Temperature (K)').grid(row = 2, column = 0)		
		self.temp_entry = Tk.Entry(self.set_up.master)
		self.temp_entry.grid(row = 2, column = 1)
		self.temp_entry.insert(0, '298') #Default entry

		#Builds a text box to input cutoff radius in angstrohms 
		self.cutoff_label = Tk.Label(self.set_up.master, text = 'Cutoff (A)').grid(row = 3, column = 0)		
		self.cutoff_entry = Tk.Entry(self.set_up.master)
		self.cutoff_entry.grid(row = 3, column = 1)
		self.cutoff_entry.insert(0, '10') #Default entry

		#Builds a drop down menu to choose 'Test' or 'Slab'
		self.function_options = ['SLAB', 'TEST']
		self.function_chosen = Tk.StringVar(self.set_up.master)
		self.function_chosen.set(self.function_options[0]) #Default value

		self.function_menu = apply(Tk.OptionMenu, (self.set_up.master, self.function_chosen) + tuple(self.function_options))
		self.function_menu.pack()
		self.function_menu.grid(column = 0, row = 4)

				
		def run_program():

			model = self.model_chosen.get()
			T = self.temp_entry.get()
			cutoff = self.cutoff_entry.get()
			func = self.function_chosen.get()
			
			
			nsite, AT, Q, M, LJ, mol_sigma = ut.get_param(model)

			#Directory information is found in
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

			#if self.model_options in ['ARGON', 'METHANOL', 'ETHANOL', 'DMSO']: root = '/local/scratch/sam5g13/AMBER/{}/T_{}_K/CUT_{}_A'.format(model, T, cutoff)
			#elif self.model_options == 'AMOEBA': root = '/local/scratch/sam5g13/OpenMM/WATER/{}/T_{}_K/CUT_{}_A'.format(model, T, cutoff)
			#else: root = '/local/scratch/sam5g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A'.format(model, T, cutoff)
			if model in ['SPCE']: root =  '/local/scratch/sam5g13/AMBER/TEST'

			if not os.path.exists(root):

				self.warning_message = tkMessageBox.showerror('Error', 'Files not found')
				exit() #CHANGE LATER TO RE-SET
		
			#print model, T , cutoff, func

			if func == 'SLAB':
				"---------------------------------------------ENTERING SLAB-----------------------------------------------------"

				import sys
				import mdtraj as md
				from scipy import constants as con

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
				#print root

				sigma = np.max(LJ[1])

				if model not in ['AMOEBA']:

					self.rad_dist = tkMessageBox.askyesno('Cubic Radial Distribution', 'Perform Radial Distribution?')

					if self.rad_dist:
						#-------------CUBIC RADIAL DISTRIBUTION-----------
	
						directory = '{}/CUBE'.format(root)

						if not os.path.exists("{}/DATA".format(directory)): os.mkdir("{}/DATA".format(directory))

						if not os.path.exists("{}/DATA/DEN".format(directory)): os.mkdir("{}/DATA/DEN".format(directory))

						#traj = ut.load_nc(root, 'CUBE', model, 'cube')
						traj = ut.load_nc(root, directory, model, 'cube')
		
						natom = int(traj.n_atoms)
						nmol = int(traj.n_residues)
						nframe = int(traj.n_frames)
						"lOADING DIMENSIONS IN NM, CONVERT TO ANGTROMS"
						DIM = np.array(traj.unitcell_lengths[0]) * 10

						lslice = 0.01
						lslice_nm = lslice / 10.
						max_r = np.min(DIM) / 20.
						nslice = int(max_r / lslice)

						"""		
						new_XYZ = np.zeros((nframe, natom, 3))
						for image in xrange(nframe):
	
		
							XYZ= np.transpose(traj.xyz[image])	
			
							xmol, ymol, zmol = ut.molecules(XYZ[0], XYZ[1], XYZ[2], nsite, M, com = "COM")

							XYZ = np.transpose([xmol, ymol, zmol])
							for i in xrange(nmol): new_XYZ[image, i] = XYZ[i]
						traj.xyz = new_XYZ
		
						pairs = []
						for i in xrange(100): 
							for j in xrange(i): pairs.append([i, j])
							#for j in xrange(i): pairs.append([i*nsite, j*nsite])
							#print pairs

						r, g_r = md.compute_rdf(traj, pairs = pairs, bin_width = lslice_nm, r_range = (0, max_r))
				
						with open('{}/DATA/DEN/{}_{}_{}_RDEN.txt'.format(directory, model.lower(), nslice, nframe), 'w') as outfile:
							np.savetxt(outfile, (r, g_r), fmt='%-12.6f')
						"""
						with open('{}/DATA/DEN/{}_{}_{}_RDEN.txt'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
							r, g_r = np.loadtxt(infile)

						fig = plt.figure(0, figsize=(6,6))
						plt.plot(r*10, g_r)

						mol_sigma = 2**(1./6) * g_r.argmax() * lslice

						self.return__rdf_info = tkMessageBox.showinfo('Cubic Radial Distribution', "r_max = {}    molecular sigma = {}".format(g_r.argmax()*lslice, mol_sigma))
						
					
						#Makes a Canvas to plot graphs on
						self.rdf_graph_canvas = FigureCanvasTkAgg(fig , master = Tk.Tk())
						self.rdf_graph_canvas.get_tk_widget().pack()
						self.rdf_graph_canvas.draw()
												

		


				"END OF RDF"
				
				directory = '{}/{}'.format(root, folder.upper())
	

				#----------BUILDING SURFACE POSITIONAL ARRAYS-----------

				self.start_pos_array = tkMessageBox.showinfo('Building Surface Positional Arrays', 'You are about to build the surface positional arrays.')
			
				if not os.path.exists("{}/DATA/POS".format(directory)): os.mkdir("{}/DATA/POS".format(directory))

				self.ow_pos = tkMessageBox.askyesno('Building Surface Positional Arrays', 'Overwrite At Mol Positions?')

				if os.path.exists('{}/DATA/parameters.txt'.format(directory)) and not self.ow_pos:
					DIM = np.zeros(3)
					with file('{}/DATA/parameters.txt'.format(directory), 'r') as infile:
						natom, nmol, nframe, DIM[0], DIM[1], DIM[2] = np.loadtxt(infile)
					natom = int(natom)
					nmol = int(nmol)
					nframe = int(nframe)

					self.return_loading_info = tkMessageBox.showinfo('Building Surface Postional Arrays', 'Files exist and overwrite is off. The Parameter and Com files are being loaded.')

					with file('{}/DATA/POS/{}_{}_COM.txt'.format(directory, model.lower(), nframe), 'r') as infile:
						cell_com = np.loadtxt(infile)

				else:
					#-----> NEED FILES TO CHECK IF THIS WORKS
					
					
					self.return_loading_info1 = tkMessageBox.showinfo('Building Surface Postional Arrays', 'Files do not exist or overwrite is on. The Parameter and Com files are being created.')

					traj = ut.load_nc(root, directory, model, suffix)
					
					#PUT IN A PROGRESS BAR	
					
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

				#-----------STARTING DENSITY PROFILE------------

				self.start_density_profile = tkMessageBox.showinfo('Density Profile', 'Starting density profile.')
	
				if not os.path.exists("{}/DATA/DEN".format(directory)): os.mkdir("{}/DATA/DEN".format(directory))

				#ADD IN TICK BOXES
				ow_all = False
				ow_count = False

				if os.path.exists('{}/DATA/DEN/{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nframe)) and not ow_all:
					self.ow_y_n_df = tkMessageBox.askyesno('Density Profile', 'The density profile file has been found, do you wish to overwrite?') 
					if bool(self.ow_y_n_df) == True:
						self.ask_ow_count = tkMessageBox.askyesno('Density Profile', 'Do you wish to overwrite the count?') 
						self.ow_count = bool(self.ask_ow_count) == True
						den.density_profile(directory, model, nframe, natom, nmol, nsite, AT, M, cell_com, DIM, nslice, ow_count)	
				else: den.density_profile(directory, model, nframe, natom, nmol, nsite, AT, M, cell_com, DIM, nslice, ow_all)
				
				return
	
				#------STARTING INTRINSIC DENSITY PROFILE-------
		
				
				self.start_intrinsic_density_profile = tkMessageBox.showinfo('Intrinsic Density Profile', 'Starting intrinsic denisty profile.')

				ow_all = False
				ow_coeff = False
				ow_curve = False
				ow_count = False
				ow_wden = False

				if not os.path.exists("{}/DATA/INTDEN".format(directory)): os.mkdir("{}/DATA/INTDEN".format(directory))
				if not os.path.exists("{}/DATA/INTPOS".format(directory)): os.mkdir("{}/DATA/INTPOS".format(directory))

				if os.path.exists('{}/DATA/INTDEN/{}_{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nm, nframe)) and not ow_all:
					self.ow_y_n_idf = tkMessageBox.askyesno('Intrinsic Density Profile', 'The intrinsic density profile file has been found, do you wish to overwrite?') 
					if bool(self.ow_y_n_idf) == True:
						self.ow_coeff = tkMessageBox.askyesno('Intrinsic Density Profile', 'Overwrite coefficients?') 
						self.ow_count = tkMessageBox.askyesno('Intrinsic Density Profile', 'Overwrite count?') 
						self.ow_wden = tkMessageBox.askyesno('Intrinsic Density Profile', 'Overwrite wden?') 
						surf.intrinsic_profile(directory, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, mol_sigma, cell_com, DIM, nslice, ncube, nm, vlim, self.ow_coeff, ow_curve, self.ow_count, self.ow_wden)
				else: surf.intrinsic_profile(directory, model, csize, suffix, nframe, natom, nmol, nsite, AT, M, mol_sigma, cell_com, DIM, nslice, ncube, nm, vlim, ow_all, ow_all, ow_all, ow_all)

				#graphs.print_graphs_density(directory, model, nsite, AT, nslice, nm, cutoff, csize, folder, suffix, nframe, DIM)

				if model != 'ARGON':

					ow_all = False
					ow_angles = False

					#--------STARTING ORIENTATIONAL PROFILE--------

					self.start_orientational_profile = tkMessageBox.showinfo('Orientational Profile', 'You are about to start the orientational profile.')

					if not os.path.exists("{}/DATA/EULER".format(directory)): os.mkdir("{}/DATA/EULER".format(directory))
					if not os.path.exists("{}/DATA/INTEULER".format(directory)): os.mkdir("{}/DATA/INTEULER".format(directory))

					if os.path.exists('{}/DATA/EULER/{}_{}_{}_{}_EUL.txt'.format(directory, model.lower(), nslice, a_type, nframe)) and not ow_all:
						self.ow_y_n_op = tkMessageBox.askyesno('Orientational Profile', 'The orientational profile file has been found, do you wish to overwrite?') 
						if bool(self.ow_y_n_op) == 'TRUE':  
							self.ow_angles = tkMessageBox.askyesno('Orientational Profile', 'Overwrite angles?') 
							self.ow_polar = tkMessageBox.askyesno('Orientational Profile', 'Overwrite polarisabilities?') 
							ori.euler_profile(directory, nframe, nslice, nmol, model, csize, suffix, AT, Q, M, LJ, cell_com, DIM, nsite, a_type, nm, self.ow_angles, self.ow_polar)
					else: ori.euler_profile(directory, nframe, nslice, nmol, model, csize, suffix, AT, Q, M, LJ, cell_com, DIM, nsite, a_type, nm, ow_all, ow_all)

				#graphs.print_graphs_orientational(directory, model, nsite, AT, nslice, nm, a_type, cutoff, csize, folder, suffix, nframe, DIM)

				ow_all = False
				ow_ecount = False
				ow_acount = False

				#-------STARTING DIELECTRIC PROFILE--------

				self.start_dielectric_profile = tkMessageBox.showinfo('Dielectric Profile', 'You are about to start the dielectric profile.')

				if not os.path.exists("{}/DATA/DIELEC".format(directory)): os.mkdir("{}/DATA/DIELEC".format(directory))
				if not os.path.exists("{}/DATA/INTDIELEC".format(directory)): os.mkdir("{}/DATA/INTDIELEC".format(directory))
				if not os.path.exists("{}/DATA/ELLIP".format(directory)): os.mkdir("{}/DATA/ELLIP".format(directory))

				if os.path.exists('{}/DATA/DIELEC/{}_{}_{}_{}_DIE.txt'.format(directory, model.lower(), nslice, a_type, nframe)) and not ow_all:
					self.ow_y_n_dp = tkMessageBox.askyesno('Dielectric Profile', 'The dielectric profile file has been found, do you wish to overwrite?') 
					if bool(self.ow_y_n_dp) == 'TRUE':   
	
						#The original code was hashed out previously,I've left it here along with the new code, also hashed out, below it. The die.dielectric function hasn't been updated to the new code.
						
						#ow_ecount = bool(raw_input("OVERWRITE ECOUNT? (Y/N): ").upper() == 'Y')
						#self.ow_ecount = tkMessageBox.askyesno('Dielectric Profile', 'Overwrite ecount?')
 
						#ow_acount = bool(raw_input("OVERWRITE ACOUNT? (Y/N): ").upper() == 'Y') 
						#self.ow_acount = tkMessageBox.askyesno('Dielectric Profile', 'Overwrite acount?') 

						die.dielectric_refractive_index(directory, model, csize, AT, sigma, nslice, nframe, a_type, nm, DIM, ow_ecount, ow_acount)
				else: die.dielectric_refractive_index(directory, model, csize, AT, sigma, nslice, nframe, a_type, nm, DIM, ow_all, ow_acount)

				#graphs.print_graphs_dielectric(directory, model, nsite, AT, nslice, nm, a_type, cutoff, csize, folder, suffix, nframe, DIM)

				#-------STARTING ELLIPSOMETRY PREDICTIONS--------

				self.start_ellipsometry_predictions = tkMessageBox.showinfo('Ellipsometry Predictions', 'You are about to start the ellipsometry predictions')

				ellips.transfer_matrix(directory, model, csize, AT, sigma, nslice, nframe, a_type, nm, DIM, cutoff)

			elif func == 'TEST':

				#---------------------------------------------ENTERING TEST-----------------------------------------------------
				

				value  = []
				TYPE = value 

				def which_variation():
		
					value.append(dictionary['which_variation'])
					self.top.destroy()
					

				def which_variation1():

					value.append(dictionary['which_variation1'])
					self.top.destroy()

				def which_variation2():

					value.append(dictionary['which_variation2'])
					self.top.destroy()

			
				dictionary = {'which_variation' : 'W', 'which_variation1' : 'A', 'which_variation2' : 'C'}

				self.top = Tk.Toplevel()
				#self.top.geometry('300x100')
				self.top.title('Entering Test')
	
				self.which_variation = Tk.Message(self.top, text = 'Width (W), Area (A) or Cubic (C) variation?', width = 300)
				self.which_variation.grid(row = 0, column = 1)

				Width = self.width_button = Tk.Button(self.top, text = 'Width', command = which_variation)
				Width.grid(row = 1, column = 0, sticky = 'w')

				Area = self.area_button = Tk.Button(self.top, text = 'Area', command = which_variation1)
				Area.grid(row = 1, column = 1)

				Cubic = self.cubic_button = Tk.Button(self.top, text = 'Cubic', command = which_variation2)
				Cubic.grid(row = 1, column = 2, sticky = 'e')
				
				self.top.wait_window()
				
				
				self.VDW_force_corrections = tkMessageBox.askyesno('VDW Force Corrections', 'Apply VDW Force Corrections?')

				root = '{}/{}_TEST'.format(root, TYPE)

				if bool(self.VDW_force_corrections) == 'TRUE': folder = 'SURFACE_2'
				else: folder = 'SURFACE' 

				suffix = 'surface'
				csize = 50

				if model == 'ARGON':
					if folder.upper() == 'SURFACE_2':
						if TYPE == ['W']: 
							nfolder = 60
							sfolder = 11 
						elif TYPE == ['A']: 
							nfolder = 25
							sfolder = 4
						elif TYPE == ['C']: 
							nfolder = 22
							sfolder = 0
					else:
						if TYPE == ['W']: 
							nfolder = 60
							sfolder = 11
						elif TYPE == ['C']: 
							nfolder = 7
							csize = 30
						elif TYPE == ['A']: self.choose_model_again = tkMessageBox.showinfo('Model Variation', 'The type of variation you have selected doesnt exist. Please try again.')

				if model == 'TIP4P2005':
					if TYPE == ['W']: 
						nfolder = 40
						sfolder = 11 
					if TYPE == ['A']: nfolder = 25
					if TYPE == ['C']:
						nfolder = 2
						csize = 35
					if T != 298: nfolder = 1	

				if model in ['SPCE', 'TIP3P']:
					if TYPE == ['W']:
						nfolder = 40
						sfolder = 11
					else: self.choose_model_again = tkMessageBox.showinfo('Model Variation', 'The type of variation you have selected doesnt exist. Please try again.')

				#print "We made it to here", TYPE, nfolder, bool(self.VDW_force_corrections)
				
				#build = bool(raw_input("Make input files or Analyse?").upper() == 'Y')
				#Don't understand the above hashed out code				

				self.build = tkMessageBox.askyesno('Make input files or Analyse', 'Do you wish to make the input files ') 
				#self.build = bool(self.make_or_analyse) == 'TRUE'
									
				print bool(self.build)

				if bool(self.build) == ['TRUE']: pass
				#Don't get why this isnt working
				else:


					

					

					value  = []
					TASK = value 

					def density_profile():
	
						value.append(dictionary['density_profile'])
						self.top1.destroy()
				

					def intrinsic_surface_profiling():

						value.append(dictionary['intrinsic_surface_profiling'])
						self.top1.destroy()

					def orientational_profile():

						value.append(dictionary['orientational_profile'])
						self.top1.destroy()

					def dielectric_refractive_index():
	
						value.append(dictionary['dielectric_refractive_index'])
						self.top1.destroy()

					def thermodynamics():

						value.append(dictionary['thermodynamics'])
						self.top1.destroy()

					def ellipsometry_module():

						value.append(dictionary['ellipsometry_module'])
						self.top1.destroy()

					def print_graphs():

						value.append(dictionary['print_graphs'])
						self.top1.destroy()

		
					dictionary = {'density_profile' : 'D', 'intrinsic_surface_profiling' : 'IS', 'orientational_profile' : 'O', 'dielectric_refractive_index' : 'E', 'thermodynamics' : 'T', 'ellipsometry_module' : 'EL', 'print_graphs' : 'G'}

					self.top1 = Tk.Toplevel()
					#self.top.geometry('300x100')
					self.top1.title('Tasks')

					self.which_variation = Tk.Message(self.top1, text = 'Density profile (D), Intrinsic Surface Profiling (A), Orientational Profile (C), Dielectric Refractive Index (E), Thermodynamics (T), Ellipsometry Module (EL), Print Graphs (G) ', width = 300)
					self.which_variation.grid(row = 0, column = 1)

					D = self.D_button = Tk.Button(self.top1, text = 'Density profile', command = density_profile)
					D.grid(row = 1, column = 0, sticky = 'w')

					A = self.A_button = Tk.Button(self.top1, text = 'Intrinsic Surface Profiling', command = intrinsic_surface_profiling)
					A.grid(row = 1, column = 1)

					C = self.C_button = Tk.Button(self.top1, text = 'Orientational Profile', command = orientational_profile)
					C.grid(row = 1, column = 2, sticky = 'e')

					E = self.E_button = Tk.Button(self.top1, text = 'Dielectric Refractive Index', command = dielectric_refractive_index)
					E.grid(row = 2, column = 1)

					T = self.T_button = Tk.Button(self.top1, text = 'Thermodynamics', command = thermodynamics)
					T.grid(row = 3, column = 0, sticky = 'e')

					EL = self.EL_button = Tk.Button(self.top1, text = 'Ellipsometry Module', command = ellipsometry_module)
					EL.grid(row = 3, column = 1)

					G = self.G_button = Tk.Button(self.top1, text = 'Print Graphs', command = print_graphs)
					G.grid(row = 3, column = 2, sticky = 'w')

					self.top1.wait_window()
			
					print TASK

					if TASK == ['D']:

						
						self.task = tkMessageBox.askyesno('Density Profile', 'Overwrite all density files?')
						self.ow_all = bool(self.task) 
						print self.ow_all
						sigma = np.max(LJ[1])
						lslice = 0.05 * sigma

						for i in xrange(sfolder, nfolder):
							root_ = '{}/{}_{}'.format(root, TYPE, i)
							directory = '{}/{}'.format(root_, folder.upper())
		
							self.ow_den = True
							self.ow_count = False

							if os.path.exists('{}/DATA/parameters.txt'.format(directory)) and self.ow_all == False:
								#self.params_load = tkMessageBox.showinfo('Denisty Profile', 'Loading {}/DATA/parameters.txt'.format(directory))
								#NEED A WINDOW THAT IS ABLE TO UPDATE QUICKLY
								print "LOADING {}/DATA/parameters.txt".format(directory)
								with file('{}/DATA/parameters.txt'.format(directory), 'r') as infile:
									_, _, nframe, _, _, dim_Z = np.loadtxt(infile)

								nframe = int(nframe)
								nslice = int(dim_Z / lslice)

								if os.path.exists('{}/DATA/DEN/{}_{}_{}_DEN.txt'.format(directory, model.lower(), nslice, nframe)) and not ow_all:
									#NEED A WINDOW THAT IS ABLE TO UPDATE QUICKLY
									print "FILE FOUND '{}/DATA/DEN/{}_{}_{}_DEN.txt".format(directory, model.lower(), nslice, nframe)
									self.ow_den = tkMessageBox.askyesno('Density Profile', 'Overwrite density?')
									self.ow_den1 = bool(self.ow_den) == 'TRUE'
									if self.ow_den: 
										self.ow_count = tkMessageBox.askyesno('Density Profile', 'Overwrite count?')
										self.ow_count1 = bool(self.ow_count) == 'TRUE'
									

							if self.ow_den or self.ow_all:
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

					elif TASK == ['T']:

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


						self.ow_area1 = tkMessageBox.askyesno('Density Profile', 'Overwrite Intrinsic Surface Area?')
						self.ow_ntb1 = tkMessageBox.askyesno('Density Profile', 'Overwrite Surface Tension Error?')
						self.ow_est1 = tkMessageBox.askyesno('Density Profile', 'Overwrite Average Energy and Tension?')
						self.ow_area = bool(self.ow_area1) == 'TRUE'
						self.ow_ntb = bool(self.ow_ntb1) == 'TRUE'
						self.ow_est = bool(self.ow_est1) == 'TRUE'

						(ENERGY, ENERGY_ERR, TEMP, TEMP_ERR, TENSION, TENSION_ERR, VAR_TENSION, N_RANGE, A_RANGE, AN_RANGE, Z_RANGE, DEN) = thermo.energy_tension(
							root, model, suffix, TYPE, folder, sfolder, nfolder, T, rc, LJ, csize, e_constant, l_constant, st_constant, com, self.ow_area, self.ow_ntb, self.ow_est)

					elif TASK == ['G']:
						import graphs
						graphs.print_graphs_thermodynamics(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, folder, suffix)
				
					print TASK, lslice

		#Button that runs the program 
		self.run_button = Tk.Button(self.set_up.master, text = 'Run', command = run_program).grid(row = 5, column = 0)
	

	
		

GUI = set_up(1)
