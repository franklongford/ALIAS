import numpy as np
import os

def write_vac(ROOT, element, cutoff, T, vdwmeth, size, k):

	if vdwmeth == 0: root = '{}/VAC'.format(ROOT)
	else:root = '{}/VAC_2'.format(ROOT)
	if not os.path.exists(root): os.mkdir(root)
	FILE = open('{}/vac_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
	FILE.write('{}: 500ps MD vacuum, cutoff = {}  \n'.format(element.upper(), cutoff))
	FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
	FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 250000, dt = 0.002,\n'.format(T,T))
	FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
	if atom == "argon": FILE.write(' &ewald\n  vdwmeth = {},\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n'.format(vdwmeth))
	else: FILE.write(' &ewald\n  fft_grids_per_ang = {},\n  vdwmeth = {}\n/\n'.format(1 + 0.1 * k, vdwmeth))
	FILE.close()

	FILE = open('{}/vac_{}_{}.pbs'.format(root, element.lower(), size), 'w')
	FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=05:00:00\n#PBS -l nodes=2:ppn=16\n\n ')
	FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
	FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
	FILE.write('mpirun -n 32 $AMBERHOME/bin/pmemd.MPI -O -i vac_{}_{}.in -o {}_{}_vac.out -p ../{}_{}.prmtop -c {}_{}_vin.rst -r {}_{}_vac.rst\n'.format(
		   element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
	FILE.close()


def write_surface(ROOT, element, cutoff, T, vdwmeth, size, k):

	if vdwmeth == 0: root = '{}/SURFACE'.format(ROOT)
	else:root = '{}/SURFACE_2'.format(ROOT)
	if not os.path.exists(root): os.mkdir(root)
	FILE = open('{}/surface_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
	FILE.write('{}: 4ns MD surface, cutoff = {}  \n'.format(element.upper(), cutoff))
	FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
	FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 2000000, dt = 0.002,\n'.format(T,T))
	FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
	if atom == "argon": FILE.write(' &ewald\n  vdwmeth = {},\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n'.format(vdwmeth))
        else: FILE.write(' &ewald\n  fft_grids_per_ang = {},\n  vdwmeth = {}\n/\n'.format(1 + 0.1 * k, vdwmeth))
	FILE.close()

	FILE = open('{}/surface_{}_{}.pbs'.format(root, element.lower(), size), 'w')
	if vdwmeth == 0: FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=20:00:00\n#PBS -l nodes=2:ppn=16\n\n ')
	else: FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=45:00:00\n#PBS -l nodes=2:ppn=16\n\n ') 
	FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
	FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
	FILE.write('mpirun -n 32 $AMBERHOME/bin/pmemd.MPI -O -i surface_{}_{}.in -o {}_{}_surface.out -p ../{}_{}.prmtop -c ../VAC_2/{}_{}_vac.rst -r {}_{}_surface.rst\n'.format(
	      element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
	FILE.close()


#UTOFF = [8]#, 9, 10, 11, 12, 18]
cutoff = 10
atom = 'TIP4P2005'
T = 298
K_RANGE = range(20)

root = '/scratch/fl7g13/WATER/{}/T_{}_K/CUT_{}_A/K_TEST'.format(atom.upper(), T, cutoff)
if not os.path.exists(root): os.mkdir(root)
size = 50

for k in K_RANGE:
	root0 = '{}/K_{}'.format(root, k)
        if not os.path.exists(root0): os.mkdir(root0)
        write_vac(root0, atom, cutoff, T, 0, size, k)
	write_surface(root0, atom, cutoff, T, 0, size, k)

