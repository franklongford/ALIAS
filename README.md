# ALIAS: Air-Liquid Interface Analysis Suite

By Frank Longford (2016)
------------------------

Intrinsic surface identifier built for Python following the Intrinsic Sampling method of 
[Chacon and Tarazona 2004](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.70.235407).
Surface pivot density optimisation routine follows guidelines based on diffusion rate by
[Duque, Tarazona and Chacon 2008](http://aip.scitation.org/doi/10.1063/1.2841128).
Extra functions included for Surface Reconstruction routine by 
[Longford, Essex, Skylaris and Frey 2018](https://aip.scitation.org/doi/10.1063/1.5055241).

Installation
------------

ALIAS requires a local distributions of `python >= 3.6` and `pip >= 9.0` in order to run.
Either [Enthought](https://www.enthought.com/product/enthought-python-distribution/),
[anaconda](https://www.anaconda.com/download/) or [miniconda](https://conda.io/miniconda.html)
distributions are recommended.

The use of a package manager, such as [edm](https://www.enthought.com/product/enthought-deployment-manager/) or
[conda](https://conda.io/docs/), is optional, but also recommended.

#### Native Installation

If you wish to install ALIAS to run in your native machine environment, you will first need to install both
`click` and `setuptools` python packages. Afterwards, run the command

    python -m ci install

and enter the installation command for your local package manager when prompted. Please run
the unit tests after installation is complete using the command

    python -m ci test

Be advised that if you are not using `edm` or `conda` package managers, there is no guarantee that all required
libraries will be available to install. In which case you may need to follow the steps below for installation in
a virtual environment.

#### EDM Installation (recommended)

A light-weight installation can be performed using the Enthought Deployment Manager (EDM). After downloading
[edm](https://www.enthought.com/product/enthought-deployment-manager/), simply create a default bootstrap environment
using:

    edm install -e bootstrap --version 3.6 -y click setuptools
    edm shell -e bootstrap

Then build the `alias-py36` environment using the following command:

    python -m ci build-env --edm

Afterwards, install a package egg with all binaries using:

    python -m ci install --edm

This will install all required libraries and create the local `ALIAS` binary.
To make sure the installation has been successful, please run the unittests

    python -m ci test --edm

#### Conda Installation

If using anaconda or miniconda python distribution, this can be easily initiated by creating a default bootstrap
environment:

    conda create -n bootstrap python=3.6 -y click setuptools
    source activate bootstrap

Then build the `alias-py36` environment using same command as before but with the `--conda` flag:

    python -m ci build-env --conda

Afterwards, activate the PyFibre environment and install a package egg with all binaries using:

    source activate alias-py36
    python -m ci install --conda

This will install all required libraries and create the local `ALIAS` binary.
To make sure the installation has been successful, please run the unittests

    python -m ci test


Instructions:
-------------

Main routine of ALIAS can be run from inside the ``alias-py36`` environment via the following commands:

1) ``ALIAS [traj] [top] [flags]``

	`traj`: Trajectory file
		
	`top`: Topology file  
		
	`flags`:
	
	    --recon      Perform surface reconstruction routine
		--ow_coeff   Overwrite existing intrnisic surface coefficients
		--ow_recon   Overwrite reconstructed surface coefficients
		--ow_intpos  Overwrite intrinsic molecular positions and derivatives
		--ow_hist    Overwrite histograms of intrinsic distributions
		--ow_dist    Overwrite average intrinsic density and curvature distributions
		
	(see [MDTraj](http://mdtraj.org/1.9.0/index.html) homepage for supported filetypes and detailed instructions)

2) *Choose residue to use for surface identification:* (optional)

	Select name of molecule / fragment / residue that will be used to map the intrinsic surface.

	If only one residue type is detected, selects by default.

3) *Use standard elemental masses?*

	Decide whether to use [MDTraj](http://mdtraj.org/1.9.0/index.html) experimental atomic mass database for each atomic site on selected residue or enter forcefield masses if known (recommended).

4) *Use atomic site as centre of molecular mass?*

	Decide whether to use an atomic site as a proxy molecular position or calculate centre of mass explicitly.

	For light molecules (water, methane etc..), using a single atomic site is usually sufficient.

5) *Enter molecular radius:*

	Enter a suitable radius `mol_sigma` (in angstroms) for the molecular interaction sphere.

	For small molecules with one LJ site, this should be the LJ sigma parameter.

	Larger molecules may require experimenation using different radii. 

	Note: radius determines minimum seperation distance between molecules, so interaction sphere should not be less than this distance.

6) *Use recommended weighting coefficient for surface area minimisation?*

	RECOMMENDED! Only change this if you know what you are doing.

7) *Manually enter in new surface pivot number? (search will commence otherwise)*

	Enter maximum number of pivots included intrinsic fitting routine for each surface. 

	If not selected, search will commence to minmise pivot diffusion rate, as recommended by 
	[Duque, Tarazona and Chacon 2008](http://aip.scitation.org/doi/10.1063/1.2841128).


File Tree:
-------------

Output of main routine will produce following file tree structure in the `traj` directory:

    alias_analysis
    │
    ├── ...chk.pkl
    │
    ├── data
    │    │
    │    ├── pos
    │    │    ├── ...xmol.npy
    │    │    ├── ...ymol.npy
    │    │    ├── ...zmol.npy
    │    │    └── ...com.npy		
    │    │
    │    ├── surface
    │    │    ├── ...coeff.hdf5
    │    │    └── ...pivot.hdf5
    │    │
    │    ├── intpos
    │    │    ├── ...int_z_mol.hdf5
    │    │    ├── ...int_dxdy_mol.hdf5
    │    │    └── ...int_ddxddy_mol.hdf5
    │    │
    │    └── intden
    │         ├── ...count_corr.hdf5
    │         └── ...int_den_curve.npy
    │     
    └── figures




