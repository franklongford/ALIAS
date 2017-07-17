import numpy as np
import os
import mdtraj as md

"""
*************** CONVERTS AMBER RESTART FILES TO ONETEP DAT FILES *******************


***************************************************************
Created 7/6/17  by Frank Longford

Last modified 7/6/17

"""

def read_positions(file_, nsite):
	"OPENS FILE AND RETURNS POSITIONS OF ATOMS AS X Y Z ARRAYS"
	FILE = open('{}'.format(file_), 'r')
	lines = FILE.readlines()
	FILE.close()

	l = len(lines)

	temp = lines[1].split()

	natom = int(temp[0])
	nmol = natom / nsite
	ndof = natom * 3

	x = []
	y = []
	z = []

	nline = int(ndof/6) + 2

	"Loops through .rst file to copy atomic positions in rows of 6 positions long"
	for i in range(l)[2:nline]:

		temp_lines = lines[i].split()
	
		x.append(float(temp_lines[0]))
		x.append(float(temp_lines[3]))

		y.append(float(temp_lines[1]))
		y.append(float(temp_lines[4]))

		z.append(float(temp_lines[2]))
		z.append(float(temp_lines[5]))
	
	"Checks if there is a half row in the .rst file (3 positions long)"
	if np.mod(ndof,6) != 0:
		
		temp_lines = lines[nline].split()
	
		x.append(float(temp_lines[0]))
		y.append(float(temp_lines[1]))
		z.append(float(temp_lines[2]))


	return x, y, z


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


def write_top(directory, FILE, species):

	with open("{}/{}.dat".format(directory, FILE), 'w') as fout:
		fout.write("# {} POLARISABILITY CALCULATION\n".format(species.upper()))


def write_instructions(directory, FILE, task, efield=[0,0,0]):

	h_plot = 0
	l_plot = 0
	maxit_ngwf = 100

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
			fout.write("\n")
		
		elif task == "SinglePoint":

			fout.write("write_xyz               : T\n")
			fout.write("cube_format             : T\n")
			fout.write("polarisation_calculate  : T\n")
			fout.write("writepolarisationplot   : T\n")
			fout.write("\n")
			fout.write("constant_efield         : {} {} {}\n".format(efield[0], efield[1], efield[2]))
			fout.write("\n")

		fout.write("ngwf_threshold_orig     : 2.0E-6\n")
		fout.write("homo_plot               : {}\n".format(h_plot))
		fout.write("lumo_plot               : {}\n".format(l_plot))
		fout.write("maxit_ngwf_cg           : {}\n".format(maxit_ngwf))
		fout.write("minit_lnv               : 10\n")
		fout.write("maxit_lnv               : 20\n")
		fout.write("write_density_plot      : T\n")
		fout.write("read_tightbox_ngwfs     : F\n")
		fout.write("\n")
		fout.write("fftbox_batch_size       : 1\n")
		fout.write("threads_max             : 1\n")
		fout.write("threads_num_fftboxes    : 1\n")
		fout.write("threads_num_mkl         : 1\n")
		fout.write("\n")


def write_block_lattice(directory, FILE):
	
	with open("{}/{}.dat".format(directory, FILE), 'a') as fout:
		fout.write("\n")
		fout.write("%BLOCK LATTICE_CART\n")
		fout.write("ang\n")

		fout.write("40.0 0.0 0.0\n")
		fout.write("0.0 40.0 0.0\n")
		fout.write("0.0 0.0 40.0\n")

		fout.write("%ENDBLOCK LATTICE_CART\n")


def write_species(directory, FILE, species):

	with open("{}/{}.dat".format(directory, FILE), 'a') as fout:
		fout.write("\n")
		fout.write("%BLOCK SPECIES\n")
		fout.write("O O 8 4 8.0\n")
		fout.write("H H 1 1 8.0\n")
		if species.lower() not in ['tip3p', 'spce', 'tip4p2005']: fout.write("C C 6 4 8.0\n")
		if species.lower() == 'dmso': fout.write("S S 16 4 8.0\n")
		fout.write("%ENDBLOCK SPECIES\n")

		fout.write("\n")
		fout.write("%BLOCK SPECIES_POT\n")
		fout.write("O '/scratch/fl7g13/ONETEP/PSEUDO/O_01PBE_OP.recpot'\n")
		fout.write("H '/scratch/fl7g13/ONETEP/PSEUDO/H_00PBE_OP.recpot'\n")
		if species.lower() not in ['tip3p', 'spce', 'tip4p2005']: fout.write("C '/scratch/fl7g13/ONETEP/PSEUDO/carbon.recpot'\n")
		if species.lower() == 'dmso': fout.write("S '/scratch/fl7g13/ONETEP/PSEUDO/s-optgga1.recpot'\n")
		fout.write("%ENDBLOCK SPECIES_POT\n")


def write_positions(directory, FILE, X, Y, Z, ATOMS):
	
	with open("{}/{}.dat".format(directory, FILE), 'a') as fout:
		fout = open("{}/{}.dat".format(directory, FILE), 'a')

		fout.write("\n")
		fout.write("%BLOCK POSITIONS_ABS\n")
		fout.write("ang\n")
		
		for i in xrange(nmol):
			index = i * nsite
			for j in xrange(nsite):
				x = X[index + j]
				y = Y[index + j]
				z = Z[index + j]
				fout.write("{} {:7.7f} {:7.7f} {:7.7f}\n".format(ATOMS[j], x, y, z))	

		fout.write("%ENDBLOCK POSITIONS_ABS\n")


def write_xyz(drectory, X, Y, Z, FILE, ATOMS):

	with open("{}/{}.xyz".format(directory, FILE), 'w') as fout:
		fout.write("{}\n".format(natom))
		fout.write("SNAPSHOT ATOMIC COORDINATES\n")

		for i in range(nmol):
			index = i * nsite
			for j in range(3):
				fout.write("{} {:7.7f} {:7.7f} {:7.7f}\n".format(ATOMS[j], X[index + j], Y[index + j], Z[index + j]))


def write_pbs(directory, FILE):

	OUT = open('{}/{}.pbs'.format(directory, FILE), 'w')
	OUT.write('#!/bin/sh\n')
	OUT.write('#PBS -S /bin/bash\n')
	OUT.write('#PBS -l nodes=1:ppn=1\n')
	OUT.write('#PBS -l walltime=8:00:00\n')
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
	OUT.write('nprocs=`echo $(($ncores / $nthreads))`\n')
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


model = raw_input("Model: ")
model = model.lower()

if model in ['tip3p', 'spce', 'tip4p2005']:
        abb = model
        ATOMS = ['O', 'H', 'H']	
elif model == 'methanol':
	abb = 'meoh'
	ATOMS = ['H', 'C', 'H', 'H', 'O', 'H']
elif model == 'ethanol':
        abb = 'eoh'
        ATOMS = ['H', 'C', 'H', 'H', 'C', 'H', 'H', 'O', 'H']
elif model == 'dmso':
        abb = 'dmso'
        ATOMS = ['H', 'C', 'H', 'H', 'S', 'O', 'C', 'H', 'H', 'H']
else:
	raise TypeError
	print "ERROR: model not in paramter list"
	sys.exit() 

nsite = len(ATOMS)
prm_file = '{}.prmtop'.format(abb.lower())
rst_file = '{}.inpcrd'.format(abb.lower())
nmol = 1
natom = nmol * nsite

ROOT = '/scratch/fl7g13/ONETEP/{}'.format(model.upper())
xat, yat, zat = read_positions('{}/{}'.format(ROOT, rst_file), nsite)

task = "GeometryOptimization"
directory = '{}/GEOM_OPT'.format(ROOT)
out = "{}_geomopt".format(abb)
if not os.path.exists(directory): os.mkdir(directory)

write_top(directory, out, model)
write_instructions(directory, out, task)
write_block_lattice(directory, out)
write_species(directory, out, model)
write_positions(directory, out, xat, yat, zat, ATOMS)
write_xyz(directory, xat, yat, zat, abb, ATOMS)
write_pbs(directory, out)

task = "SinglePoint"
FOLDERS = ['LOCAL_FIELD', 'EX', 'EY', 'EZ']
field_strength = 0.1

FIELD = [[0.0, 0.0, 0.0],
	 [field_strength, 0.0, 0.0],
	 [0.0, field_strength, 0.0],
	 [0.0, 0.0, field_strength]] 

for n, folder in enumerate(FOLDERS):
	directory = '{}/{}'.format(ROOT, folder)

	task = "SinglePoint"
	if folder == 'LOCAL_FIELD': out = "{}_singlepoint".format(abb)
	else: out = "{}_singlepoint_{}".format(abb, folder.lower())

	if not os.path.exists(directory): os.mkdir(directory)
	
	write_top(directory, out, model)
	write_instructions(directory, out, task, efield=FIELD[n])
	write_block_lattice(directory, out)
	write_species(directory, out, model)
	write_positions(directory, out, xat, yat, zat, ATOMS)
	write_xyz(directory, xat, yat, zat, abb, ATOMS)
	write_pbs(directory, out)

