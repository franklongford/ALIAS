import numpy as np
import os
import mdtraj as md

"""
*************** CONVERTS AMBER RESTART FILES TO ONETEP DAT FILES *******************

PROGRAM INPUT:  f = NAME OF RESTART FILES TO BE PROCESSED
		t = ONETEP TASK (SINGLE POINT OR GEOM OPTIMISE)	       

PROGRAM OUTPUT: 'onetep_(s)_(t).dat' = ONETEP DAT FILE


***************************************************************
Created July 2014 by Frank Longford

Last modified 28/8/14

"""

def move_atoms(position, edge, index):

	rank = index % 3
	diff = edge - position[index]	
	diff += 0.05 * np.sign(diff)
	if rank == 0:
		position[index] += diff 
		position[index+1] += diff
		position[index+2] += diff
	elif rank == 1:
		position[index] += diff 
		position[index-1] += diff
		position[index+1] += diff
	else:
		position[index] += diff 
		position[index-1] += diff
		position[index-2] += diff

	return position


def write_top(directory, FILE):

	fout = open("{}/{}.dat".format(directory, FILE), 'w')

	fout.write("# WATER SURFACE CALCULATION\n")

	fout.close()


def write_instructions(directory, FILE, task):

	if task == "GeometryOptimization":
                h_plot = 0
                l_plot = 0
                maxit_ngwf = 100
	elif task == "SinglePoint":
		h_plot = 0
		l_plot = 0
		maxit_ngwf = 100
	elif task == "Cond":
		h_plot = -1
                l_plot = -1
		maxit_ngwf = 200

	with open("{}/{}.dat".format(directory, FILE), 'a') as fout:
		fout.write("TASK                    : {}\n".format(task))
		fout.write("CUTOFF_ENERGY           : 800 eV\n")
		fout.write("output_detail           : VERBOSE\n")
		fout.write("do_properties           : T\n")
		fout.write("xc_functional           : PBE\n")
		fout.write("maxit_hotelling         : 100\n")
		fout.write("odd_psinc_grid          : T\n")

		if task == "GeometryOptimization": 
			fout.write("lnv_threshold_orig      : 1.0e-9\n")
			fout.write("write_denskern          : T\n")
			fout.write("write_tightbox_ngwfs    : T\n")
			fout.write("write_converged_dkngwfs : T\n")
			fout.write("ngwf_cg_max_step        : 4\n")
			fout.write("lnv_cg_max_step         : 4\n")

			fout.write("geom_disp_tol           : 1e-2 bohr\n")
			fout.write("geom_force_tol          : 2e-2 \"ha/bohr\"\n")
			fout.write("geom_maxiter            : 40\n")
	
		if task == "SinglePoint":
			fout.write("write_xyz               : T\n")
                        fout.write("cube_format             : T\n")
                        fout.write("polarisation_calculate  : T\n")
                        fout.write("writepolarisationplot   : T\n")
		if task == "Cond":
                        fout.write("\n")
                        fout.write("cond_num_states           : 8\n")
                        fout.write("cond_num_extra_states     : 5\n")
                        fout.write("cond_num_extra_its        : 15\n")
                        fout.write("cond_calc_max_eigen       : T\n")
                        fout.write("read_denskern             : T\n")
                        fout.write("read_tightbox_ngwfs       : T\n")
                        fout.write("cond_plot_vc_orbitals     : F\n")
                        fout.write("cond_minit_lnv            : 15\n")
                        fout.write("cond_maxit_lnv            : 20\n")
                        fout.write("cond_calc_optical_spectra : T\n")
                        fout.write("cond_spec_print_mat_els   : F\n")
			fout.write("maxit_pen                 : 0\n")

		if task == 'LR_TDDFT':
			fout.write("read_denskern                  : T\n")
			fout.write("read_tightbox_ngwfs            : T\n")
			fout.write("cond_read_tightbox_ngwfs       : T\n")
			fout.write("lr_tddft_num_states            : 1\n")
			fout.write("lr_tddft_projector             : T\n")
			fout.write("lr_tddft_joint_set             : T\n")
			fout.write("lr_tddft_write_densities       : F\n")
			fout.write("lr_tddft_analysis              : T\n")
			fout.write("lr_tddft_precond               : T\n")
			fout.write("lr_tddft_write_kernels         : T\n")
			fout.write("lr_tddft_restart               : F\n")
		else:
                        fout.write("\n")
			fout.write("ngwf_threshold_orig     : 2.0E-6\n")
                        fout.write("homo_plot               : {}\n".format(h_plot))
                        fout.write("lumo_plot               : {}\n".format(l_plot))
                        fout.write("maxit_ngwf_cg           : {}\n".format(maxit_ngwf))
                        fout.write("minit_lnv               : 10\n")
                        fout.write("maxit_lnv               : 20\n")
                        fout.write("write_density_plot      : F\n")

		fout.write("\n")
		fout.write("fftbox_batch_size       : 8\n")
		fout.write("threads_max             : 64\n")
		fout.write("threads_num_fftboxes    : 4\n")
		fout.write("threads_num_mkl         : 4\n")


def write_block_lattice(directory, dim, FILE):
	
	fout = open("{}/{}.dat".format(directory, FILE), 'a')

	fout.write("\n")
	fout.write("%BLOCK LATTICE_CART\n")
	fout.write("ang\n")

	fout.write("{:7.7f} 0.0 0.0\n".format(dim[0]))
	fout.write("0.0 {:7.7f} 0.0\n".format(dim[1]))
	fout.write("0.0 0.0 {:7.7f}\n".format(dim[2]))

	#fout.write("15.0 0.0 0.0\n".format(dim[0]))
        #fout.write("0.0 15.0 0.0\n".format(dim[1]))
        #fout.write("0.0 0.0 27.0\n".format(dim[2]))

	fout.write("%ENDBLOCK LATTICE_CART\n")

	fout.close()


def write_species(directory, FILE, task, n_ngwfs, cutoff):

	fout = open("{}/{}.dat".format(directory, FILE), 'a')

	fout.write("\n")
	fout.write("%BLOCK SPECIES\n")
	fout.write("O O 8 4 8.0\n")
	fout.write("H H 1 1 8.0\n")
	fout.write("%ENDBLOCK SPECIES\n")

	if task == "Cond" or task == 'LR_TDDFT':
		fout.write("\n")
		fout.write("%BLOCK SPECIES_COND\n")
		fout.write("O O 8 {} {}\n".format(n_ngwfs[0], cutoff))
		fout.write("H H 1 {} {}\n".format(n_ngwfs[1], cutoff))
		fout.write("%ENDBLOCK SPECIES_COND\n")

	fout.write("\n")
	fout.write("%BLOCK SPECIES_POT\n")
	fout.write("O '/scratch/fl7g13/ONETEP/PSEUDO/O_01PBE_OP.recpot'\n")
	fout.write("H '/scratch/fl7g13/ONETEP/PSEUDO/H_00PBE_OP.recpot'\n")
	fout.write("%ENDBLOCK SPECIES_POT\n")

	fout.close()


def write_positions(directory, X, Y, Z, FILE):
	
	fout = open("{}/{}.dat".format(directory, FILE), 'a')

	fout.write("\n")
	fout.write("%BLOCK POSITIONS_ABS\n")
	fout.write("ang\n")
	
	for i in range(nmol):
		index = i * nsite
		for j in range(3):
			xat = X[index + j] - int(X[index + j] / DIM[0]) * DIM[0]
			yat = Y[index + j] - int(Y[index + j] / DIM[1]) * DIM[1]
			zat = Z[index + j] - int(Z[index + j] / DIM[2]) * DIM[2]
			fout.write("{} {:7.7f} {:7.7f} {:7.7f}\n".format(ATOMS[j], xat, yat, zat))	

	fout.write("%ENDBLOCK POSITIONS_ABS\n")

	fout.close()


def write_xyz(drectory, X, Y, Z, FILE):

	fout = open("{}/{}.xyz".format(directory, FILE), 'a')

	fout.write("{}\n".format(natom))
	fout.write("SNAPSHOT ATOMIC COORDINATES\n")

	for i in range(nmol):
		index = i * nsite
		for j in range(3):
			fout.write("{} {:7.7f} {:7.7f} {:7.7f}\n".format(ATOMS[j], X[index + j], Y[index + j], Z[index + j]))
	fout.close()


def write_pbs(directory, FILE):

	OUT = open('{}/{}.pbs'.format(directory, FILE), 'w')
	OUT.write('#!/bin/sh\n')
	OUT.write('#PBS -S /bin/bash\n')
	OUT.write('#PBS -l nodes=4,tpn=16\n')
	OUT.write('#PBS -l walltime=24:00:00\n')
	OUT.write('#PBS -N {}\n'.format(FILE))
	OUT.write('\n')

	OUT.write('# load modules\n')
	OUT.write('module purge\n')
	OUT.write('module load torque moab\n')
	OUT.write('module load intel/2016 intel/mkl/2016 intel/tbb/2016 intel/ipp/2016 intel/vtune/2016 intel/advisor/2016 intel/inspector/2016 intel/mpi/2016 intel/parallelxe/2016\n')
	OUT.write('\n')

	OUT.write('cd $PBS_O_WORKDIR\n')
	OUT.write('\n')

	OUT.write('# Set up threading and machinefile\n')
	OUT.write('cat $PBS_NODEFILE | uniq > nodes\n')
	OUT.write("ncores=`wc -l $PBS_NODEFILE | awk '{{ print $1 }}'`\n")
	OUT.write("nthreads=`grep -m1 threads_max $PBS_O_WORKDIR/{}.dat | awk '{{ print $3 }}'`\n".format(FILE))
	OUT.write('nprocs=`echo $(($ncores * $nthreads))`\n')
	OUT.write('\n')

	OUT.write('export "OMP_NUM_THREADS=1"\n')
	OUT.write('ulimit -s unlimited\n')
	OUT.write('\n')

	OUT.write('# Output execution information to output file\n')
	OUT.write('touch {}.out\n'.format(FILE))
	OUT.write('cat nodes >> {}.out\n'.format(FILE))
	OUT.write('echo "Processes: " $nprocs "Threads: " $nthreads "Cores: " $ncores >> {}.out\n'.format(FILE))
	OUT.write('echo "mpirun -n $nprocs -machinefile nodes /scratch/fl7g13/ONETEP_4.5.8.19/main/bin/onetep.iridis4.intel16.omp {}.dat 2> {}.err" >> {}.out\n'.format(FILE, FILE, FILE))
	OUT.write('\n')

	OUT.write('mpirun -np $nprocs -machinefile nodes -bootstrap rsh /scratch/fl7g13/ONETEP_4.5.8.19/main/bin/onetep.iridis4.intel16.omp {}.dat >> {}.out 2> {}.err\n'.format(FILE, FILE, FILE))
	OUT.close()


"""
model = raw_input("Model: ")
nsite = raw_input("Nsites: ")
csize = int(raw_input("Cubic cell dimension: "))
traj_file = raw_input("Trajectory file: ")
prm_file = raw_input("Parameter file: ")
"""

model = 'methanol'
nsite = 4
asize = 50
zsize = 50
prm_file = '{}.prmtop'.format(model.lower())

ATOMS = ['O', 'H', 'H']
#FOLDERS = ['CUBE', 'SURFACE']

"""
for folder in FOLDERS:
	traj_file = '{}_{}_{}.nc'.format(model.lower(), zsize, folder.lower())
	ROOT = '/scratch/fl7g13/ONETEP/{}/{}_{}'.format(model.upper(), asize, zsize)
	traj = md.load('{}/{}/{}'.format(ROOT, folder, traj_file), top='{}/{}'.format(ROOT, prm_file))
	natom = traj.n_atoms
	nmol = traj.n_residues
	ntraj = traj.n_frames

	root = '{}/{}'.format(ROOT, folder)
	tasks = ['G', 'S', 'C', 'T']

	#t = raw_input("Task: SINGLE(S), GEOM(G), CONDUCTION(C), TDDFT(T)? ")

	DIM = np.array(traj.unitcell_lengths[0]) * 10

	ZYX = np.rot90(traj.xyz[-1])
	zat = ZYX[0] * 10
	yat = ZYX[1] * 10
	xat = ZYX[2] * 10

	for t in tasks:
		if t.upper() == "G":
			task = "GeometryOptimization"
			out = "water_{}_geomopt".format(zsize)
			directory = '{}/GEOMETRY'.format(root)
		elif t.upper() == "S":
			task = "SinglePoint"
			out = "water_{}_singlepoint".format(zsize)
			directory = '{}/SINGLE_POINT'.format(root)
		elif t.upper() == "C":
			task = "Cond"
			out = "water_{}_conduction".format(zsize)
			directory = '{}/CONDUCTION'.format(root)
		elif t.upper() == "T":
			task = "LR_TDDFT"
			out = "water_{}_tddft".format(zsize)
			directory = '{}/TDDFT'.format(root)
		
		if not os.path.exists(directory): os.mkdir(directory)
		write_top(directory, out)
		write_instructions(directory, out, task)
		write_block_lattice(directory, DIM, out)
		write_species(directory, out, task, [5, 13], 12.0)
		write_positions(directory, xat, yat, zat, out)
		write_xyz(directory, xat, yat, zat, 'water_{}_{}'.format(zsize, t))
		write_pbs(directory, out)
"""
