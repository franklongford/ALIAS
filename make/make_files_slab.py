import numpy as np
import os

def write_eq1(ROOT, size, element, cutoff, ntc, T, nodes, ppn):

	root = '{}/EQ1'.format(ROOT, cutoff, element.upper(), size)
	if not os.path.exists(root): os.mkdir(root)

	if ntc == 1: dt = 0.001
	else: dt = 0.002

	time = 100
	nstlim = int(time / dt)

	FILE = open('{}/eq1_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
	FILE.write('{}: 100ps MD with equilibration T 0-->298K, cutoff = {}  \n'.format(element.upper(), cutoff))
	FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 0,\n  ntx    = 1,\n  ntb    = 1,\n  cut    = {},\n  ntc    = {},\n  ntf    = {},\n'.format(float(cutoff), ntc, ntc))
	FILE.write('  tempi  = 0.0,\n  temp0  = {},\n  ntt    = 3,\n  gamma_ln   = 10.0,\n  nstlim = {}, dt = {},\n'.format(T, nstlim, dt))
	FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n  nmropt = 1,\n/\n')
	FILE.write(' &ewald\n  vdwmeth = 0 \n/\n')
	FILE.write(' &wt TYPE=TEMP0, ISTEP1=1, ISTEP2={}, VALUE1=0, VALUE2={}\n /\n'.format(int(4./5 * nstlim), T))
	FILE.write(' &wt TYPE=TEMP0, ISTEP1={}, ISTEP2={}, VALUE1={}, VALUE2={}\n /\n'.format(int(4/5. * nstlim) + 1, nstlim, T, T))
	FILE.write(' &wt TYPE=END\n / \n/')
	FILE.close()

	FILE = open('{}/eq1_{}_{}.pbs'.format(root, element.lower(), size), 'w')
	FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=2:00:00\n#PBS -l nodes={}:ppn={}\n\n '.format(nodes, ppn))
	FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
	FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
	FILE.write('mpirun -n {} $AMBERHOME/bin/pmemd.MPI -O -i eq1_{}_{}.in -o {}_{}_eq1.out -p ../{}_{}.prmtop -c ../{}_{}.inpcrd -r {}_{}_eq1.rst\n'.
		   format(nodes*ppn, element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
	FILE.close()


def write_eq2(ROOT, size, element, cutoff, ntc, T, nodes, ppn):

	root = '{}/EQ2'.format(ROOT, cutoff, element.upper(), size)
	if not os.path.exists(root): os.mkdir(root)

	if ntc == 1: dt = 0.001
        else: dt = 0.002

	time = 500
        nstlim = int(time / dt)

	FILE = open('{}/eq2_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
	FILE.write('{}: 500ps MD, cutoff = {}  \n'.format(element.upper(), cutoff))
	FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 2,\n  ntp    = 1,\n  pres0  = 1,\n  cut    = {},\n  ntc    = {},\n  ntf    = {},\n'.format(float(cutoff), ntc, ntc))
	FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  gamma_ln   = 10.0,  taup = 2,\n  nstlim = {}, dt = {},\n'.format(T,T, nstlim, dt))
	FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
	FILE.write(' &ewald\n  vdwmeth = 0 \n/\n')
	FILE.close()

	FILE = open('{}/eq2_{}_{}.pbs'.format(root, element.lower(), size), 'w')
	FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=6:00:00\n#PBS -l nodes={}:ppn={}\n\n '.format(nodes, ppn))
	FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
	FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
	FILE.write('mpirun -n {} $AMBERHOME/bin/pmemd.MPI -O -i eq2_{}_{}.in -o {}_{}_eq2.out -p ../{}_{}.prmtop -c ../EQ1/{}_{}_eq1.rst -r {}_{}_eq2.rst\n'.format(
		nodes*ppn, element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
	FILE.close()


def write_vac(ROOT, size, element, cutoff, ntc, T, vdwmeth,  nodes, ppn):

	if vdwmeth == 0: root = '{}/VAC'.format(ROOT)
	else:root = '{}/VAC_2'.format(ROOT)
	if not os.path.exists(root): os.mkdir(root)

	if ntc == 1: dt = 0.001
        else: dt = 0.002

	time = 2000
        nstlim = int(time / dt)

	FILE = open('{}/vac_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
	FILE.write('{}: 2ns MD vacuum, cutoff = {}  \n'.format(element.upper(), cutoff))
	FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = {},\n  ntf    = {},\n'.format(float(cutoff), ntc, ntc))
	FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = {}, dt = {},\n'.format(T,T, nstlim, dt))
	FILE.write('  ntpr = 10, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
	FILE.write(' &ewald\n  fft_grids_per_ang = 2,\n  vdwmeth = {}\n/\n'.format(vdwmeth))
	FILE.close()

	FILE = open('{}/vac_{}_{}.pbs'.format(root, element.lower(), size), 'w')
	FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=04:00:00\n#PBS -l nodes={}:ppn={}\n\n '.format(nodes, ppn))
	FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
	FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
	FILE.write('mpirun -n {} $AMBERHOME/bin/pmemd.MPI -O -i vac_{}_{}.in -o {}_{}_vac.out -p ../{}_{}.prmtop -c {}_{}_vin.rst -r {}_{}_vac.rst\n'.format(
		   int(nodes*ppn), element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
	FILE.close()


def write_surface(ROOT, size, element, cutoff, ntc, T, vdwmeth, nodes, ppn):

	if vdwmeth == 0: root = '{}/SURFACE'.format(ROOT)
	else:root = '{}/SURFACE_2'.format(ROOT)
	if not os.path.exists(root): os.mkdir(root)

	if ntc == 1: dt = 0.001
        else: dt = 0.002

	time = 8000
        nstlim = int(time / dt)

	FILE = open('{}/surface_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
	FILE.write('{}: 8ns MD surface, cutoff = {}  \n'.format(element.upper(), cutoff))
	FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = {},\n  ntf    = {},\n'.format(float(cutoff), ntc, ntc))
	FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = {}, dt = {},\n'.format(T,T,nstlim,dt))
	FILE.write('  ntpr = 10, ntwx = 1000, ntwv = -1, ntwr = 5000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
	FILE.write(' &ewald\n  fft_grids_per_ang = 2,\n  vdwmeth = {}\n/\n'.format(vdwmeth))
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


def write_cube(ROOT, size, element, cutoff, T, nodes, ppn):

        root = '{}/CUBE'.format(ROOT)
        if not os.path.exists(root): os.mkdir(root)

        FILE = open('{}/cube_{}_{}.in'.format(root, element.lower(), cutoff), 'w')
        FILE.write('{}: 4ns MD cube, cutoff = {}  \n'.format(element.upper(), cutoff))
        FILE.write(' &cntrl\n  imin   = 0,\n  irest  = 1,\n  ntx    = 5,\n  ntb    = 1,\n  cut    = {},\n  ntc    = 1,\n  ntf    = 2,\n'.format(float(cutoff)))
        FILE.write('  tempi  = {},\n  temp0  = {},\n  ntt    = 3,\n  ene_avg_sampling = 1,\n  gamma_ln   = 10.0,\n  nstlim = 2000000, dt = {},\n'.format(T,T,dt))
        FILE.write('  ntpr = 1000, ntwx = 1000, ntwr = 10000,\n  ioutfm = 1, iwrap = 1, ig = -1,\n /\n')
        if atom == "argon": FILE.write(' &ewald\n  vdwmeth = 0,\n  nfft1 = 6,\n  nfft2 = 6,\n  nfft3 = 6\n/\n')
        else: FILE.write(' &ewald\n  vdwmeth = 0\n/\n')
        FILE.close()

        FILE = open('{}/cube_{}_{}.pbs'.format(root, element.lower(), size), 'w')
        FILE.write('#!/bin/bash\n#PBS -S /bin/bash\n#PBS -l walltime=06:00:00\n#PBS -l nodes={}:ppn={}\n\n '.format(nodes, ppn))
        FILE.write('export AMBERHOME=/home/fl7g13/amber12\nmodule load intel\n')
        FILE.write('module load openmpi/1.6.4/intel\ncd $PBS_O_WORKDIR\n\n')
        FILE.write('mpirun -n {} $AMBERHOME/bin/pmemd.MPI -O -i cube_{}_{}.in -o {}_{}_cube.out -p ../{}_{}.prmtop -c ../EQ2/{}_{}_eq2.rst -r {}_{}_cube.rst -x {}_{}_cube.nc\n'.format(
              nodes * ppn, element.lower(), cutoff, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size, element.lower(), size))
        FILE.close()


def write_cmd(root, size, element, ntc):

	if element == 'tip3p': abb = 'TIP3PBOX'
	elif element == 'tip4p': abb = 'T4P'
	elif element == 'tip4p2005': abb = 'T4P2005'
	elif element == 'methanol': abb = 'MEOH'
	elif element == 'ethanol': abb = 'EOH'
	elif element == 'dmso': abb = 'DMSO'
	elif element == 'spcefw': abb = 'SPF'
	elif element == 'argon': abb = 'ARGN'
	else: abb = element.upper()

	FILE = open('{}/{}_{}.cmd'.format(root, element.lower(), size), 'w')
	FILE.write('clearVariables\nlogFile {}_{}.log\n'.format(element.lower(), size))
	FILE.write('#\n#        Make water ball\n#\n')
	FILE.write('Source /home/fl7g13/amber12/dat/leap/cmd/leaprc.ff03.r2\n')
	if element != 'tip3p': 
		FILE.write('frcmod{} = loadamberparams frcmod.{}\n'.format(element, element))
		if element not in ['argon', 'methanol', 'dmso', 'ethanol']: 
			FILE.write('HOH = {}\n'.format(abb))
			FILE.write('WAT = {}\n'.format(abb))
		elif element in ['ethanol', 'methanol', 'dmso']: FILE.write('loadamberprep {}.in\n'.format(abb.lower()))
	FILE.write('slab = createunit "{}_{}"\n'.format(element.lower(), size))
	if ntc == 1: FILE.write('set default FlexibleWater on\n')
	FILE.write('solvatebox slab {} {{{} {} {}}}\n'.format(abb, size/2.0,size/2.0, 25))
	FILE.write('saveamberparm slab {}_{}.prmtop {}_{}.inpcrd\nquit'.format(element.lower(), size, element.lower(), size))
	FILE.close()


model = 'DMSO'
atom = model.lower()
ntc = 2

if atom in ['methanol', 'dmso', 'ethanol']: 
	root = '/scratch/fl7g13/{}'.format(model)
	cutoff = 22
	vdwmeth = 0

elif atom == 'argon':
	root = '/scratch/fl7g13/ARGON'
        cutoff = 10
        vdwmeth = 3
else: 
	root = '/scratch/fl7g13/WATER/{}'.format(atom.upper())
	cutoff = 10
	vdwmeth = 3

if not os.path.exists(root): os.mkdir(root)

if atom == "argon": T = 85
else: T = 298

root = '{}/T_{}_K'.format(root, T)
if not os.path.exists(root): os.mkdir(root)
root = '{}/CUT_{}_A'.format(root,cutoff)
if not os.path.exists(root): os.mkdir(root)
root = '{}/SLAB'.format(root)
if not os.path.exists(root): os.mkdir(root)

start = 80
ndim = 1

write_cmd(root, start, atom, ntc)
write_eq1(root, start, atom, cutoff, ntc, T, 1, 16)
write_eq2(root, start, atom, cutoff, ntc, T, 1, 16)
#write_cube(root, start, atom, cutoff, T, 1, 16)
write_vac(root, start, atom, cutoff, ntc, T, vdwmeth, 4, 16)
write_surface(root, start, atom, cutoff, ntc, T, vdwmeth, 4, 16)

