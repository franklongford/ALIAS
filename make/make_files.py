import numpy as np
import os

def write_eq1(ROOT, element, cutoff, T, vdwmeth, start, ndim):

	for i in xrange(ndim):
		size = i * 5 + start
		root = '{}/CUT_{}_A/{}_{}/EQ1'.format(ROOT, cutoff, element.upper(), size)
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
		root = '{}/CUT_{}_A/{}_{}/EQ2'.format(ROOT, cutoff, element.upper(), size)
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

def write_cube(ROOT, element, cutoff, T, start, ndim):

        #for i in xrange(ndim):
        size = 50#i * 5 + start
        root = '{}/CUT_{}_A/{}_{}/CUBE'.format(ROOT, cutoff, element.upper(), size)
        FILE = open('{}/cube_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
        FILE.write('{}: 4ns MD cube, cutoff = {}  \n'.format(element.upper(), cutoff))
        FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
        FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 2000000, dt = 0.002,\n'.format(T,T))
        FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
        if atom == "argon": FILE.write(' &ewald\n  vdwmeth = 0,\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n')
        else: FILE.write(' &ewald\n  vdwmeth = 0\n/\n')
        FILE.close()

        FILE = open('{}/cube_{}_{}.pbs'.format(root, element.lower(), size), 'w')
        FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=05:00:00\n#PBS -l nodes=2:ppn=16\n\n ')
        FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
        FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
        FILE.write('mpirun -n 32 $AMBERHOME/bin/pmemd.MPI -O -i cube_{}_{}.in -o {}_{}_cube.out -p ../{}_{}.prmtop -c ../EQ2/{}_{}_eq2.rst -r {}_{}_cube.rst\n'.format(
  	      element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
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
		if atom == "argon": FILE.write(' &ewald\n  vdwmeth = {},\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n'.format(vdwmeth))
                else: FILE.write(' &ewald\n  vdwmeth = {}\n/\n'.format(vdwmeth))
		FILE.close()

		FILE = open('{}/vac_{}_{}.pbs'.format(root, element.lower(), size), 'w')
		FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=30:00:00\n#PBS -l nodes=2:ppn=16\n\n ')
		FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
		FILE.write('mpirun -n 32 $AMBERHOME/bin/pmemd.MPI -O -i vac_{}_{}.in -o {}_{}_vac.out -p ../{}_{}.prmtop -c {}_{}_vac.rst -r {}_{}_vac.rst\n'.format(
			   element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
		FILE.close()

def write_surface(ROOT, element, cutoff, T, vdwmeth, start, ndim):

	for i in xrange(ndim):
		size = i * 5 + start
		if vdwmeth == 0: root = '{}/CUT_{}_A/{}_{}/SURFACE'.format(ROOT, cutoff, element.upper(), size)
		else:root = '{}/CUT_{}_A/{}_{}/SURFACE_2'.format(ROOT, cutoff, element.upper(), size)
		FILE = open('{}/surface_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
		FILE.write('{}: 10ps MD surface, cutoff = {}  \n'.format(element.upper(), cutoff))
		FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
		FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 5000, dt = 0.002,\n'.format(T,T))
		FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 5000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
		if element.lower() == "argon": FILE.write(' &ewald\n  vdwmeth = {},\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n'.format(vdwmeth))
		else: FILE.write(' &ewald\n  vdwmeth = {}\n/\n'.format(vdwmeth))
		FILE.close()

		FILE = open('{}/surface_{}_{}.pbs'.format(root, element.lower(), size), 'w')
		FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=60:00:00\n#PBS -l nodes=2:ppn=16\n\n ')
		FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
		FILE.write('for i in {0..399}\ndo\n	')
		FILE.write('mpirun -n 32 $AMBERHOME/bin/pmemd.MPI -O -i surface_{}_{}.in -o {}_{}_surface$i.out -p ../{}_{}.prmtop -c {}_{}_vac.rst -r {}_{}_surface$i.rst\n'.format(element.lower(), cutoff,
				element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
		FILE.write('	cp {}_{}_surface$i.rst {}_{}_vac.rst\n'.format(element.lower(), size, element.lower(), size))
		FILE.write('done')
		FILE.close()

def write_cmd(ROOT, element, cutoff, start, ndim):

	for i in xrange(ndim):
                size = i * 5 + start
		root = '{}/CUT_{}_A/{}_{}'.format(ROOT, cutoff, element.upper(), size)
		FILE = open('{}/{}_{}.cmd'.format(root, element.lower(), size), 'w')
		FILE.write('clearVariables\nlogFile {}_{}.log\n'.format(element.lower(), size))
		FILE.write('#\n#	Make water slab\n#\n')
		FILE.write('Source /home/fl7g13/amber12/dat/leap/cmd/leaprc.ff03.r2\n')
		FILE.write('slab = createunit "{}_{}"\n'.format(element.lower(), size))
		if element.lower() == 'methanol':FILE.write('solvatebox slab MEOHBOX {{{} {} {}}}\n'.format(size/2.0,size/2.0,size/2.0))
		elif element.lower() == 'argon':FILE.write('solvatebox slab ARGN {{{} {} {}}}\n'.format(size/2.0,size/2.0,size/2.0))
		elif element.lower() == 'tip3p':FILE.write('solvatebox slab TIP3PBOX {{{} {} {}}}\n'.format(size/2.0,size/2.0,size/2.0))
		elif element.lower() == 'tip4p':FILE.write('solvatebox slab TIP4PBOX {{{} {} {}}}\n'.format(size/2.0,size/2.0,size/2.0))
		elif element.lower() == 'tip5p':FILE.write('solvatebox slab TP5 {{{} {} {}}}\n'.format(size/2.0,size/2.0,size/2.0))
		else:FILE.write('solvatebox slab {} {{25.5 25.5 {}}}\n'.format(element.upper(), size/2.0,size/2.0,size/2.0))
		FILE.write('saveamberparm slab {}_{}.prmtop {}_{}.inpcrd\nquit'.format(element.lower(), size, element.lower(), size))
		FILE.close()		

CUTOFF = [18]#, 9, 10, 11, 12, 18]

TYPE = raw_input("Which atom type? ")

T = int(raw_input("What Temperature? "))

atom = TYPE.lower()
if atom.lower() == "argon" or atom.lower() == "methanol":
	#root = '/scratch/fl7g13/{}'.format(atom.upper()) 
	root = '/scratch/fl7g13/{}/T_{}_K'.format(atom.upper(), T)
	
else: 
	#root = '/scratch/fl7g13/WATER/{}'.format(atom.upper())
	root = '/scratch/fl7g13/WATER/{}/T_{}_K'.format(atom.upper(), T)

start = 50
ndim = 14

for cutoff in CUTOFF:
	#write_cmd(root, atom, cutoff, start, ndim)
	#write_eq1(root, atom, cutoff, T, 0, start, ndim)
	#write_eq2(root, atom, cutoff, T, 0, start, ndim)
	#write_vac(root, atom, cutoff, T, 0, start, ndim)
        #write_cube(root, atom, cutoff, T, start, ndim)
        write_vac(root, atom, cutoff, T, 3, start, ndim)
	#write_surface(root, atom, cutoff, T, 0, start, ndim)
        write_surface(root, atom, cutoff, T, 3, start, ndim)
