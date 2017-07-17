import numpy as np
import os

def write_eq1(ROOT, element, cutoff, T, start, ndim, nodes, ppn):

        for i in xrange(ndim):
                size = i * 5 + start

                root = '{}/EQ1'.format(ROOT)
		if not os.path.exists(root): os.mkdir(root)

                FILE = open('{}/eq1_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
                FILE.write('{}: 100ps MD with equilibration T 0-->298K, cutoff = {}  \n'.format(element.upper(), cutoff))
                FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 0,\n  ntx    = 1,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
                FILE.write('  tempi  = 0.0,\n  temp0  = {},\n  ntt    = 3,\n  gamma_ln   = 10.0,\n  nstlim = 50000, dt = 0.002,\n'.format(T))
                FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n  nmropt = 1,\n/\n')
                #if element.lower() == "argon": FILE.write(' &ewald\n  vdwmeth=0, \n nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n')
		#else: FILE.write(' &ewald\n  vdwmeth = 0 \n/\n')
                FILE.write(' &ewald\n  vdwmeth = 0 \n/\n')
		FILE.write(' &wt TYPE=TEMP0, ISTEP1=1, ISTEP2=40000, VALUE1=0, VALUE2={}\n /\n'.format(T))
                FILE.write(' &wt TYPE=TEMP0, ISTEP1=40001, ISTEP2=50000, VALUE1={}, VALUE2={}\n /\n'.format(T,T))
                FILE.write(' &wt TYPE=END\n / \n/')
                FILE.close()

                FILE = open('{}/eq1_{}_{}.pbs'.format(root, element.lower(), size), 'w')
                FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=0:20:00\n#PBS -l nodes={}:ppn={}\n\n '.format(nodes, ppn))
                FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
                FILE.write('mpirun -n {} $AMBERHOME/bin/pmemd.MPI -O -i eq1_{}_{}.in -o {}_{}_eq1.out -p ../{}_{}.prmtop -c ../{}_{}.inpcrd -r {}_{}_eq1.rst\n'.
                           format(nodes*ppn, element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
                FILE.close()


def write_eq2(ROOT, element, cutoff, T,  start, ndim, nodes, ppn):

        for i in xrange(ndim):
                size = i * 5 + start

		root = '{}/EQ2'.format(ROOT)
                if not os.path.exists(root): os.mkdir(root)

                FILE = open('{}/eq2_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
                FILE.write('{}: 500ps MD, cutoff = {}  \n'.format(element.upper(), cutoff))
                FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 2,\n  ntp    = 1,\n  pres0  = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
                FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  gamma_ln   = 10.0,  taup = 2,\n  nstlim = 250000, dt = 0.002,\n'.format(T,T))
                FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
		#if element.lower() == "argon": FILE.write(' &ewald\n  vdwmeth=0, \n nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n')
                #else: FILE.write(' &ewald\n  vdwmeth = 0 \n/\n')
		FILE.write(' &ewald\n  vdwmeth = 0 \n/\n')
                FILE.close()

                FILE = open('{}/eq2_{}_{}.pbs'.format(root, element.lower(), size), 'w')
                FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=00:30:00\n#PBS -l nodes={}:ppn={}\n\n '.format(nodes, ppn))
                FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
                FILE.write('mpirun -n {} $AMBERHOME/bin/pmemd.MPI -O -i eq2_{}_{}.in -o {}_{}_eq2.out -p ../{}_{}.prmtop -c ../EQ1/{}_{}_eq1.rst -r {}_{}_eq2.rst\n'.format(
			    nodes*ppn, element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
                FILE.close()



def write_vac(ROOT, element, cutoff, T, vdwmeth, start, ndim, nodes, ppn):

	for i in xrange(ndim):
		size = i * 5 + start

		if vdwmeth == 0: root = '{}/VAC'.format(ROOT)
                else:root = '{}/VAC_2'.format(ROOT)
                if not os.path.exists(root): os.mkdir(root)

		FILE = open('{}/vac_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
		FILE.write('{}: 500ps MD vacuum, cutoff = {}  \n'.format(element.upper(), cutoff))
		FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
		FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 250000, dt = 0.002,\n'.format(T,T))
		FILE.write('  ntpr = 10, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
		#if atom == "argon": FILE.write(' &ewald\n  vdwmeth = {},\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n'.format(vdwmeth))
                #else: FILE.write(' &ewald\n  vdwmeth = {}\n/\n'.format(vdwmeth))
		FILE.write(' &ewald\n  vdwmeth = {}\n/\n'.format(vdwmeth))
		FILE.close()

		FILE = open('{}/vac_{}_{}.pbs'.format(root, element.lower(), size), 'w')
		FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=2:00:00\n#PBS -l nodes={}:ppn={}\n\n '.format(nodes, ppn))
		FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
		FILE.write('mpirun -n {} $AMBERHOME/bin/pmemd.MPI -O -i vac_{}_{}.in -o {}_{}_vac.out -p ../{}_{}.prmtop -c {}_{}_vin.rst -r {}_{}_vac.rst\n'.format(
			   nodes*ppn, element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
		FILE.close()


def write_surface(ROOT, element, cutoff, T, vdwmeth, start, ndim, nodes, ppn):

        for i in xrange(ndim):
        	size = i * 5 + start

		if vdwmeth == 0: root = '{}/SURFACE'.format(ROOT)
                else:root = '{}/SURFACE_2'.format(ROOT)
		if not os.path.exists(root): os.mkdir(root)

        	FILE = open('{}/surface_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
        	FILE.write('{}: 8ns MD surface, cutoff = {}  \n'.format(element.upper(), cutoff))
        	FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
        	FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 4000000, dt = 0.002,\n'.format(T,T))
        	FILE.write('  ntpr = 10, ntwx = 1000, ntwv = -1, ntwr = 5000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
        	#if atom == "argon": FILE.write(' &ewald\n  vdwmeth = {},\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n'.format(vdwmeth))
        	FILE.write(' &ewald\n  vdwmeth = {}\n/\n'.format(vdwmeth))
        	#else: FILE.write(' &ewald\n  fft_grids_per_ang = 2,\n  vdwmeth = {}\n/\n'.format(vdwmeth))
		FILE.close()

		FILE = open('{}/surface_{}_{}.pbs'.format(root, element.lower(), size), 'w')
                FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=60:00:00\n#PBS -l nodes={}:ppn={}\n\n '.format(nodes, ppn))
                FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
                FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
                if vdwmeth == 0: FILE.write('mpirun -n {} $AMBERHOME/bin/pmemd.MPI -O -i surface_{}_{}.in -o {}_{}_surface.out -p ../{}_{}.prmtop -c ../VAC/{}_{}_vac.rst -r {}_{}_surface.rst -x {}_{}_surface.nc\n'.format(
                        nodes*ppn, element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
                else: FILE.write('mpirun -n {} $AMBERHOME/bin/pmemd.MPI -O -i surface_{}_{}.in -o {}_{}_surface.out -p ../{}_{}.prmtop -c ../VAC_2/{}_{}_vac.rst -r {}_{}_surface.rst -x {}_{}_surface.nc\n'.format(
                        nodes*ppn, element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))		
	
                FILE.close()

		"""
        	FILE = open('{}/surface_{}_{}.pbs'.format(root, element.lower(), size), 'w')
        	FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=06:00:00\n#PBS -l nodes={}:ppn={}\n\n '.format(nodes, ppn))
        	FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
        	FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
        	FILE.write('mpirun -n {} $AMBERHOME/bin/pmemd.MPI -O -i surface_{}_{}.in -o {}_{}_surface.out -p ../{}_{}.prmtop -c {}_{}_vac.rst -r {}_{}_surface.rst\n'.format(
        	           nodes*ppn, element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
        	FILE.close()
		"""

def write_cube(ROOT, element, cutoff, T, start, ndim):

        #for i in xrange(ndim):
        size = 50#i * 5 + start

	root = '{}/CUBE'.format(ROOT)
        if not os.path.exists(root): os.mkdir(root)
        FILE = open('{}/cube_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
        FILE.write('{}: 4ns MD cube, cutoff = {}  \n'.format(element.upper(), cutoff))
        FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 2,\n  ntf    = 2,\n'.format(float(cutoff)))
        FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 2000000, dt = 0.002,\n'.format(T,T))
        FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
        if atom == "argon": FILE.write(' &ewald\n  vdwmeth = 0,\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n')
        else: FILE.write(' &ewald\n  vdwmeth = 0\n/\n')
        FILE.close()

        FILE = open('{}/cube_{}_{}.pbs'.format(root, element.lower(), size), 'w')
        FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=04:00:00\n#PBS -l nodes=2:ppn=16\n\n ')
        FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
        FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
        FILE.write('mpirun -n 32 $AMBERHOME/bin/pmemd.MPI -O -i cube_{}_{}.in -o {}_{}_cube.out -p ../{}_{}.prmtop -c ../EQ2/{}_{}_eq2.rst -r {}_{}_cube.rst\n'.format(
              element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
        FILE.close()


def write_cmd(ROOT, element, cutoff, start, ndim, w):

	w = 0.8 + 0.05 * w
	#print w

        size = 50
        FILE = open('{}/{}_{}.cmd'.format(ROOT, element.lower(), size), 'w')
        FILE.write('clearVariables\nlogFile {}_{}.log\n'.format(element.lower(), size))
        FILE.write('#\n#        Make water slab\n#\n')
        FILE.write('Source /home/fl7g13/amber12/dat/leap/cmd/leaprc.ff03.r2\n')
        FILE.write('slab = createunit "{}_{}"\n'.format(element.lower(), size))
        if element.lower() == 'methanol':FILE.write('solvatebox slab MEOHBOX {{{} {} {}}}\n'.format(size/2.0,size/2.0,size/2.0))
        elif element.lower() == 'argon':FILE.write('solvatebox slab ARGN {{{} {} {}}}\n'.format(size*w/2.0,size*w/2.0,size/2.0))
        elif element.lower() == 'tip3p':FILE.write('solvatebox slab TIP3PBOX {{{} {} {}}}\n'.format(size/2.0,size/2.0,size/2.0))
        elif element.lower() == 'tip4p':FILE.write('solvatebox slab TIP4PBOX {{{} {} {}}}\n'.format(size/2.0,size/2.0,size/2.0))
        elif element.lower() == 'tip5p':FILE.write('solvatebox slab TP5 {{{} {} {}}}\n'.format(size/2.0,size/2.0,size/2.0))
        else:FILE.write('solvatebox slab {} {{{} {} {}}}\n'.format(element.upper(), size*w/2.0,size*w/2.0,size/2.0))
        FILE.write('saveamberparm slab {}_{}.prmtop {}_{}.inpcrd\nquit'.format(element.lower(), size, element.lower(), size))
        FILE.close()



CUTOFF = [8]#, 9, 10, 11, 12, 18]
TYPE = 'argon'
T = 85
A_RANGE = range(13, 25)

atom = TYPE.lower()
if atom.lower() == "argon" or atom.lower() == "methanol":
	#root = '/scratch/fl7g13/{}'.format(atom.upper())
	ROOT = '/scratch/fl7g13/{}'.format(atom.upper())
	if not os.path.exists(ROOT): os.mkdir(ROOT)
	ROOT = '{}/T_{}_K'.format(ROOT, T)
	if not os.path.exists(ROOT): os.mkdir(ROOT)
else: 
	#root = '/scratch/fl7g13/WATER/{}'.format(atom.upper())
	ROOT = '/scratch/fl7g13/WATER/{}/T_{}_K'.format(atom.upper(), T)

start = 50
ndim = 1

for cutoff in CUTOFF:
	ROOT = '{}/CUT_{}_A'.format(ROOT, cutoff)
        if not os.path.exists(ROOT): os.mkdir(ROOT)
	ROOT = '{}/A_TEST'.format(ROOT)
	if not os.path.exists(ROOT): os.mkdir(ROOT)
	for a in A_RANGE:
		root = '{}/A_{}'.format(ROOT, a) 
		if not os.path.exists(root): os.mkdir(root)
		#write_cmd(root, atom, cutoff, start, ndim, a)
		#write_eq1(root, atom, cutoff, T, start, ndim, 1, 8)
        	#write_eq2(root, atom, cutoff, T, start, ndim, 1, 8)
		#write_cube(root, atom, cutoff, T, start, ndim)
		#write_vac(root, atom, cutoff, T, 0, start, ndim, 1, 8)
	        #write_vac(root, atom, cutoff, T, 3, start, ndim, 1, 8)
		#write_surface(root, atom, cutoff, T, 0, start, ndim, 1, 8)
		write_surface(root, atom, cutoff, T, 3, start, ndim, 1, 8)

