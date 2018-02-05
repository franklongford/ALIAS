 ___________________________________________
|                   __ __             ____  |
|     /\     |        |       /\     /      |
|    /  \    |        |      /  \    \___   |
|   /___ \   |        |     /___ \       \  |
|  /      \  |____  __|__  /      \  ____/  |
|___________________________________________|  v1.1

    Air-Liquid Interface Analysis Suite
    By Frank Longford

Instructions:

1) run alias.sh [traj] [top]

	traj: 	Trajectory file (filetypes supported: http://mdtraj.org/1.9.0/load_functions.html#format-specific-loading-functions)
	top:	Topology file  (see MDTraj homepage: http://mdtraj.org/1.9.0/index.html for detailed instructions)


2) (optional) Choose residue to use for surface identification

	Select name of molecule / fragment / residue that will be used to map the intrinsic surface.
	If only one residue type is detected, selects by default.

3) Use elemental masses found in checkfile?

	Decide whether to use MDTraj experimental atomic mass database for each atomic site on selected residue or enter forcefield masses if known (recommended).

4) Use atomic site as centre of molecular mass?

	Decide whether to use an atomic site as a proxy molecular position or calculate centre of mass explicitly.
	For light molecules (water, methane etc..), using a single atomic site is usually sufficient.

5) Enter molecular radius:

	Enter a suitable radius (in angstroms) for the molecular interaction sphere.
	For small molecules with one LJ site, this should be the LJ sigma parameter.
	Larger molecules may require experimenation using different radii. 
	Note: radius determines minimum seperation distance between molecules, so interaction sphere should not be less than this distance.

6) Use recommended weighting coefficient for surface area minimisation?

	RECOMMENDED! Only change this if you know what you are doing.





