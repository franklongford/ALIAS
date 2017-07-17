import numpy as np
import os

def write_eq1(ROOT, element, cutoff, T, vdwmeth, start, ndim):

	for i in xrange(ndim):
		size = i * 5 + start
		if vdwmeth == 0: root = '{}/CUT_{}_A/{}_{}/EQ1'.format(ROOT, cutoff, element.upper(), size)
		else: root = '{}/CUT_{}_A/{}_{}/EQ1_2'.format(ROOT, cutoff, element.upper(), size)
		FILE = open('{}/eq1_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
		FILE.write('{}: 100ps MD with equilibration T 0-->298K, cutoff = {}  \n'.format(element.upper(), cutoff))
		FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 0,\n  ntx    = 1,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
		FILE.write('  tempi  = 0.0,\n  temp0  = {},\n  ntt    = 3,\n  gamma_ln   = 10.0,\n  nstlim = 50000, dt = 0.002,\n'.format(T))
		FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n  nmropt = 1,\n/\n')
		FILE.write(' &ewald\n  vdwmeth = 0 \n/\n')
		FILE.write(' &wt TYPE=TEMP0, ISTEP1=1, ISTEP2=40000, VALUE1=0, VALUE2={}\n /\n'.format(T))
		FILE.write(' &wt TYPE=TEMP0, ISTEP1=40001, ISTEP2=50000, VALUE1={}, VALUE2={}\n /\n'.format(T,T))
		FILE.write(' &wt TYPE=END\n / \n/')
		FILE.close()

		FILE = open('{}/eq1_{}_{}.pbs'.format(root, element.lower(), size), 'w')
                FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=12:00:00\n#PBS -l nodes=1:ppn=16\n\n ')
                FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
                FILE.write('mpirun -n 16 $AMBERHOME/bin/pmemd.MPI -O -i eq1_{}_{}.in -o {}_{}_eq1.out -p ../{}_{}.prmtop -c ../{}_{}.inpcrd -r {}_{}_eq1.rst\n'.
                           format(element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
                FILE.close()

def write_eq2(ROOT, element, cutoff, T, vdwmeth, start, ndim):

	for i in xrange(ndim):
		size = i * 5 + start
		if vdwmeth == 0: root = '{}/CUT_{}_A/{}_{}/EQ2'.format(ROOT, cutoff, element.upper(), size)
		else: root = '{}/CUT_{}_A/{}_{}/EQ2_2'.format(ROOT, cutoff, element.upper(), size)
		FILE = open('{}/eq2_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
		FILE.write('{}: 500ps MD, cutoff = {}  \n'.format(element.upper(), cutoff))
		FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 2,\n  ntp    = 1,\n  pres0  = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
		FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  gamma_ln   = 10.0,  taup = 2,\n  nstlim = 250000, dt = 0.002,\n'.format(T,T))
		FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
		FILE.write(' &ewald\n  vdwmeth = 0 \n/\n')
		FILE.close()

		FILE = open('{}/eq2_{}_{}.pbs'.format(root, element.lower(), size), 'w')
                FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=12:00:00\n#PBS -l nodes=1:ppn=16\n\n ')
                FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
                FILE.write('mpirun -n 16 $AMBERHOME/bin/pmemd.MPI -O -i eq2_{}_{}.in -o {}_{}_eq2.out -p ../{}_{}.prmtop -c ../EQ1/{}_{}_eq1.rst -r {}_{}_eq2.rst\n'.format(element.lower(), cutoff,
                                element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
                FILE.close()

def write_vac(ROOT, element, cutoff, T, vdwmeth, start, ndim):

        for i in xrange(ndim):
                size = i * 5 + start
                if vdwmeth == 0: root = '{}/CUT_{}_A/{}_{}/VAC'.format(ROOT, cutoff, element.upper(), size)
                else:root = '{}/CUT_{}_A/{}_{}/VAC_2'.format(ROOT, cutoff, element.upper(), size)
                FILE = open('{}/vac_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
                FILE.write('{}: 500ps MD vacuum, cutoff = {}  \n'.format(element.upper(), cutoff))
                FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
                FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 250000, dt = 0.002,\n'.format(T,T))
                FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
                if atom == "argon": FILE.write(' &ewald\n  vdwmeth = {}, \n nfft1 = 6\n, nfft2 = 6\n, nfft3 = 6\n/\n'.format(vdwmeth))
                else: FILE.write(' &ewald\n  vdwmeth = {}\n/\n'.format(vdwmeth))
                FILE.close()

                FILE = open('{}/vac_{}_{}.pbs'.format(root, element.lower(), size), 'w')
                FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=60:00:00\n#PBS -l nodes=2:ppn=16\n\n ')
                FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
                FILE.write('mpirun -n 32 $AMBERHOME/bin/pmemd.MPI -O -i vac_{}_{}.in -o {}_{}_vac.out -p ../{}_{}.prmtop -c {}_{}_vin.rst -r {}_{}_vac.rst\n'.format(
                           element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
                FILE.close()



def write_bubble(ROOT, element, cutoff, T, vdwmeth, start, ndim):

	for i in xrange(ndim):
		size = i * 5 + start
		if vdwmeth == 0: root = '{}/CUT_{}_A/{}_{}/BUBBLE'.format(ROOT, cutoff, element.upper(), size)
		else:root = '{}/CUT_{}_A/{}_{}/BUBBLE_2'.format(ROOT, cutoff, element.upper(), size)

		FILE = open('{}/bubble_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
		FILE.write('{}: 10ps MD surface, cutoff = {}  \n'.format(element.upper(), cutoff))
		FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
		FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n gamma_ln   = 10.0,\n nstlim = 5000, dt = 0.002,\n'.format(T,T))
		FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 5000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
		if atom == "argon": FILE.write(' &ewald\n  vdwmeth = {}, \n nfft1 = 6\n, nfft2 = 6\n, nfft3 = 6\n/\n'.format(vdwmeth))
		else: FILE.write(' &ewald\n  vdwmeth = {} \n /\n'.format(vdwmeth))
		FILE.close()

		FILE = open('{}/bubble_{}_{}.pbs'.format(root, element.lower(), size), 'w')
                FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=60:00:00\n#PBS -l nodes=2:ppn=16\n\n ')
                FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
                FILE.write('for i in {0..100}\ndo\n    ')
                FILE.write('mpirun -n 32 $AMBERHOME/bin/pmemd.MPI -O -i bubble_{}_{}.in -o {}_{}_bubble$i.out -p ../{}_{}_bubble.prmtop -c {}_{}_bin.rst -r {}_{}_bubble$i.rst -x {}_{}_bubble$i.mdcrd\n'.format(element.lower(), cutoff,
                                element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
                FILE.write('    cp {}_{}_bubble$i.rst {}_{}_bin.rst\n'.format(element.lower(), size, element.lower(), size))
                FILE.write('done')
                FILE.close()


def write_cmd(ROOT, element, cutoff, start, ndim):

        for i in xrange(ndim):
                size = i * 5 + start
                root = '{}/CUT_{}_A/{}_{}'.format(ROOT, cutoff, element.upper(), size)
                FILE = open('{}/{}_{}.cmd'.format(root, element.lower(), size), 'w')
                FILE.write('clearVariables\nlogFile {}_{}.log\n'.format(element.lower(), size))
                FILE.write('#\n#        Make water ball\n#\n')
                FILE.write('Source /home/fl7g13/amber12/dat/leap/cmd/leaprc.ff03.r2\n')
                FILE.write('ball = createunit "{}_{}"\n'.format(element.lower(), size))
                FILE.write('solvatebox ball {} {{{} {} {}}}\n'.format(element.upper(), size/2.0,size/2.0,size/2.0))
                FILE.write('saveamberparm ball {}_{}.prmtop {}_{}.inpcrd\nquit'.format(element.lower(), size, element.lower(), size))
                FILE.close()


CUTOFF = [8]#8, 9, 10, 11, 12]

TYPE = raw_input("Which atom type? ")

atom = TYPE.lower()
root = '/scratch/fl7g13/NANOBUBBLES/{}'.format(atom.upper())
if atom == "argon": T = 85
else: T = 298

start = 150
ndim = 1

vdwmeth = 0

for cutoff in CUTOFF:
        write_cmd(root, atom, cutoff, start, ndim)
	write_eq1(root, atom, cutoff, T, 0, start, ndim)
        write_eq2(root, atom, cutoff, T, 0, start, ndim)
        write_bubble(root, atom, cutoff, T, 0, start, ndim)
