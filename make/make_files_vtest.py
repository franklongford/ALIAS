import numpy as np
import os

def write_vac(ROOT, element, cutoff, T, vdwmeth, start, ndim, v):

	for i in xrange(ndim):
		size = i * 5 + start
		if not os.path.exists('{}/CUT_{}_A/{}_{}/V_TEST'.format(ROOT, cutoff, element.upper(), size)): os.mkdir('{}/CUT_{}_A/{}_{}/V_TEST'.format(ROOT, cutoff, element.upper(), size))
		if vdwmeth == 0: root = '{}/CUT_{}_A/{}_{}/V_TEST/VAC_{}'.format(ROOT, cutoff, element.upper(), size, v)
		else:root = '{}/CUT_{}_A/{}_{}/V_TEST/VAC_2_{}'.format(ROOT, cutoff, element.upper(), size, v)
		if not os.path.exists(root): os.mkdir(root)
		FILE = open('{}/vac_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
		FILE.write('{}: 500ps MD vacuum, cutoff = {}  \n'.format(element.upper(), cutoff))
		FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
		FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 250000, dt = 0.002,\n'.format(T,T))
		FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
		if atom == "argon": FILE.write(' &ewald\n  vdwmeth = {},\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n'.format(vdwmeth))
                else: FILE.write(' &ewald\n  vdwmeth = {}\n/\n'.format(vdwmeth))
		FILE.close()

		FILE = open('{}/vac_{}_{}.pbs'.format(root, element.lower(), size), 'w')
		FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=01:30:00\n#PBS -l nodes=2:ppn=16\n\n ')
		FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
		FILE.write('mpirun -n 32 $AMBERHOME/bin/pmemd.MPI -O -i vac_{}_{}.in -o {}_{}_vac.out -p ../../{}_{}.prmtop -c {}_{}_vin.rst -r {}_{}_vac.rst\n'.format(
			   element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
		FILE.close()


def write_surface(ROOT, element, cutoff, T, vdwmeth, start, ndim, v):

        for i in xrange(ndim):
        	size = i * 5 + start
		if vdwmeth == 0: root = '{}/CUT_{}_A/{}_{}/V_TEST/SURFACE_{}'.format(ROOT, cutoff, element.upper(), size, v)
                else:root = '{}/CUT_{}_A/{}_{}/V_TEST/SURFACE_2_{}'.format(ROOT, cutoff, element.upper(), size, v)
		if not os.path.exists(root): os.mkdir(root)
        	FILE = open('{}/surface_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
        	FILE.write('{}: 4ns MD surface, cutoff = {}  \n'.format(element.upper(), cutoff))
        	FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
        	FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 2000000, dt = 0.002,\n'.format(T,T))
        	FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
        	if atom == "argon": FILE.write(' &ewald\n  vdwmeth = {},\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n'.format(vdwmeth))
        	else: FILE.write(' &ewald\n  vdwmeth = {}\n/\n'.format(vdwmeth))
        	FILE.close()

        	FILE = open('{}/surface_{}_{}.pbs'.format(root, element.lower(), size), 'w')
        	if vdwmeth == 0: FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=05:00:00\n#PBS -l nodes=2:ppn=16\n\n ')
        	else: FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=10:00:00\n#PBS -l nodes=2:ppn=16\n\n ') 
		FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
        	FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
        	FILE.write('mpirun -n 32 $AMBERHOME/bin/pmemd.MPI -O -i surface_{}_{}.in -o {}_{}_surface.out -p ../../{}_{}.prmtop -c {}_{}_vac.rst -r {}_{}_surface.rst\n'.format(
        	      element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
        	FILE.close()


CUTOFF = [8]#, 9, 10, 11, 12, 18]
TYPE = 'TIP4P2005'
T = 298
V_RANGE = range(20)

atom = TYPE.lower()
if atom.lower() == "argon" or atom.lower() == "methanol":
	#root = '/scratch/fl7g13/{}'.format(atom.upper()) 
	root = '/scratch/fl7g13/{}/T_{}_K'.format(atom.upper(), T)
	
else: 
	#root = '/scratch/fl7g13/WATER/{}'.format(atom.upper())
	root = '/scratch/fl7g13/WATER/{}/T_{}_K'.format(atom.upper(), T)

start = 50
ndim = 1

for cutoff in CUTOFF:
	for v in V_RANGE:
		write_vac(root, atom, cutoff, T, 0, start, ndim, v)
	        write_vac(root, atom, cutoff, T, 3, start, ndim, v)
		write_surface(root, atom, cutoff, T, 0, start, ndim, v)
		write_surface(root, atom, cutoff, T, 3, start, ndim, v)

