==========================================	
ALIAS: Air-Liquid Interface Analysis Suite
==========================================

By Frank Longford (2016)
----------------------

Intrinsic surface identifier built for Python following the Intrinsic Sampling method of `Chacon and Tarazona 2004`_.
Surface pivot density optimisation routine follows guidelines based on diffusion rate by `Duque, Tarazona and Chacon 2008`_.
Extra functions included for Surface Reconstruction routine by Longford 2018.

.. _Chacon and Tarazona 2004: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.70.235407
.. _Duque, Tarazona and Chacon 2008: http://aip.scitation.org/doi/10.1063/1.2841128


Installation:
-------------

``python setup.py install``


Instructions:
-------------

1) ``./alias.sh [traj] [top] [flags]``

	traj: 	
		Trajectory file
	top:	
		Topology file  
	flags:
		-recon      Perform surface reconstruction routine
		-ow_coeff   Overwrite existing intrnisic surface coefficients
		-ow_recon   Overwrite reconstructed surface coefficients
		-ow_intpos  Overwrite intrinsic molecular positions and derivatives
		-ow_hist    Overwrite histograms of intrinsic distributions
		
	(see MDTraj_ homepage for supported filetypes and detailed instructions)

.. _MDTraj: http://mdtraj.org/1.9.0/index.html

2) *Choose residue to use for surface identification:* (optional)

	Select name of molecule / fragment / residue that will be used to map the intrinsic surface.

	If only one residue type is detected, selects by default.

3) *Use standard elemental masses?*

	Decide whether to use MDTraj_ experimental atomic mass database for each atomic site on selected residue or enter forcefield masses if known (recommended).

4) *Use atomic site as centre of molecular mass?*

	Decide whether to use an atomic site as a proxy molecular position or calculate centre of mass explicitly.

	For light molecules (water, methane etc..), using a single atomic site is usually sufficient.

5) *Enter molecular radius:*

	Enter a suitable radius :math:`$\sigma$` (in angstroms) for the molecular interaction sphere.

	For small molecules with one LJ site, this should be the LJ sigma parameter.

	Larger molecules may require experimenation using different radii. 

	Note: radius determines minimum seperation distance between molecules, so interaction sphere should not be less than this distance.

6) *Use recommended weighting coefficient for surface area minimisation?*

	RECOMMENDED! Only change this if you know what you are doing.

7) *Manually enter in new surface pivot number? (search will commence otherwise)*

	Enter maximum number of pivots included intrinsic fitting routine for each surface. 

	If not selected, search will commence to minmise pivot diffusion rate, as recommended by `Duque, Tarazona and Chacon 2008`_.


Output of main routine will produce following file tree structure:

::

    project
    ├── demo.py
    ├── LICENCE.txt
    ├── processes          
    │   ├── area.py
    │   └── bboxinout.py
    ├── pywps.cfg          
    ├── requirements.txt
    ├── server.py          
    ├── setup.py
    ├── static
    ├── templates
    └── tests


Extra Functions:
----------------




