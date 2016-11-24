"""
*************** MAIN INTERFACE MODULE *******************

Main program, ALIAS: Air-Liquid Interface Analysis Suite

***********************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 24/11/2016 by Frank Longford
"""

import numpy as np
import os

import utilities as ut

print ' '+ '_' * 43
print "|                   __ __             ____  |"
print "|     /\     |        |       /\     /      |" 
print "|    /  \    |        |      /  \    \___   |"
print "|   /___ \   |        |     /___ \       \  |"
print "|  /      \  |____  __|__  /      \  ____/  |"
print '|'+ '_' * 43 + '|' + '  v0.01'
print ""
print "    Air-Liquid Interface Analysis Suite"
print ""
model = raw_input("Which model? (TIP4P2005, Argon): ")

nsite, AT, Q, M, LJ = ut.get_param(model)

T = int(raw_input("Temperature: (K) "))
cutoff = int(raw_input("Cutoff: (A) "))
TYPE = raw_input("Area (A) or Width (W) variation: ")
force = raw_input("VDW Force corrections? (Y/N): ")
if force.upper() == 'Y': folder = 'SURFACE_2'
else: folder = 'SURFACE' 
suffix = 'surface'
csize = 50

ntraj = int(raw_input("Number of trajectories: "))

if model.upper() == 'ARGON' or model.upper() == 'METHANOL': root = '/data/fl7g13/AMBER/{}/T_{}_K/CUT_{}_A/{}_{}/{}_TEST'.format(model.upper(), T, cutoff, model.upper(), csize, TYPE.upper())
else: root = '/data/fl7g13/AMBER/WATER/{}/T_{}_K/CUT_{}_A/{}_{}/{}_TEST'.format(model.upper(), T, cutoff, model.upper(), csize, TYPE.upper())

print root

if model.upper() == 'ARGON':
	if TYPE.upper() == 'A': nfolder = 25 
	if TYPE.upper() == 'W': nfolder = 60
if model.upper() == 'TIP4P2005':
	if TYPE.upper() == 'A': nfolder = 25 
	if TYPE.upper() == 'W': nfolder = 47	

print ""
TASK = raw_input("What task to perform?\nD  = Density Profile\nIS = Intrinsic Surface Profiling\nO  = Orientational Profile\nE  = Dielectric and Refractive Index.\nT  = Thermodynamics\nEL = Ellipsometry module\nG  = Print Graphs\n")
print ""

if TASK.upper() == 'D': from density import main
elif TASK.upper() == 'IS': from intrinsic_surface import main
elif TASK.upper() == 'O': from orientational import main
elif TASK.upper() == 'E': from dielectric import main
elif TASK.upper() == 'T': from thermodynamics import main

main(root, model, nsite, AT, Q, M, LJ, T, cutoff, csize, TYPE, folder, nfolder, suffix, ntraj)

"""
elif TASK.upper() == 'SE':
	import surface_energy
	surface_energy.main()

if TASK.upper() == 'G':
	import graphs
	graphs.main()

if TASK.upper() == 'EL':
	import ellipsometry
	ellipsometry.main('test')
"""



