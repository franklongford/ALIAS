"""
*************** ORIENTATIONAL ANALYSIS MODULE *******************

Calculates Euler angle profile and polarisability of system

***************************************************************
Created 24/11/2016 by Frank Longford

Contributors: Frank Longford

Last modified 22/08/2017 by Frank Longford
"""

import numpy as np
import scipy as sp
import subprocess, time, sys, os, math, copy
import matplotlib.pyplot as plt

from scipy import stats
from scipy import constants as con
from scipy.optimize import curve_fit, leastsq
import scipy.integrate as spin
from scipy.interpolate import bisplrep, bisplev, splprep, splev

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import utilities as ut
import intrinsic_sampling_method as ism

def R_tensors(directory, frame, nslice, nmol, model, csize, COM, DIM, nsite):

        temp_O = np.zeros((nslice, 9))

        xat, yat, zat = ut.read_atom_positions(directory, model, csize, frame)
        xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)
        xR, yR, zR = COM[frame]

        for j in xrange(nmol):
                sys.stdout.write("PROCESSING {} ODIST {}: {} out of {}  molecules\r".format(directory, frame, j, nmol))
                sys.stdout.flush()

                molecule = np.zeros((nsite, 3))

                for l in xrange(nsite):
                        molecule[l][0] = xat[j*nsite+l]
                        molecule[l][1] = yat[j*nsite+l]
                        molecule[l][2] = zat[j*nsite+l]

                z = zmol[j] - zR

                """NORMAL Z AXIS"""

		O = ut.local_frame_molecule(molecule, model)
                if O[2][2] < -1: O[2][2] = -1.0
                elif O[2][2] > 1: O[2][2] = 1.0

                index1 = int((z + DIM[2]/2) * nslice / DIM[2]) % nslice

                for k in xrange(3):
                        for l in xrange(3):
                                index2 = k * 3 + l
                                temp_O[index1][index2] += O[k][l]**2

        with file('{}/EULER/{}_{}_{}_ODIST.npy'.format(directory, model.lower(), nslice, frame), 'w') as outfile:
                np.save(outfile, temp_O)

        return temp_O


def mol_angles(directory, frame, nslice, nmol, model, csize, COM, DIM, nsite):

        z_array = np.zeros(nmol)
        theta = np.zeros(nmol)
        phi = np.zeros(nmol)
        varphi = np.zeros(nmol)

        xat, yat, zat = ut.read_atom_positions(directory, model, csize, frame)
        xmol, ymol, zmol = ut.read_mol_positions(directory, model, csize, frame)
        xR, yR, zR = COM[frame]

	for j in xrange(nmol):
                sys.stdout.write("PROCESSING {} ANGLES {}:  {} out of {}  molecules\r".format(directory, frame, j, nmol) )
                sys.stdout.flush()

                molecule = np.zeros((nsite, 3))

                for l in xrange(nsite):
                        molecule[l][0] = xat[j*nsite+l]
                        molecule[l][1] = yat[j*nsite+l]
                        molecule[l][2] = zat[j*nsite+l]

                z = zmol[j] - zR

                """NORMAL Z AXIS"""

                O = ut.local_frame_molecule(molecule, model)

                z_array[j] = z
                if O[2][2] < -1: O[2][2] = -1.0
                elif O[2][2] > 1: O[2][2] = 1.0
                theta[j] = np.arccos(O[2][2])
                phi[j] = (np.arctan(-O[2][0] / O[2][1]))
                varphi[j] =  (np.arctan(O[0][2] / O[1][2]))

        with file('{}/EULER/{}_{}_ANGLE.npy'.format(directory, model.lower(), frame), 'w') as outfile:
                np.save(outfile, (z_array, theta, phi, varphi))

        return z_array, theta, phi, varphi


def angle_dist(directory, model, csize, nframe, nmol, nsite, COM, DIM, nslice, npi, ow_angles):

        print "BUILDING ANGLE DISTRIBUTIONS"

        dpi = np.pi / npi
        P_z_theta_phi = np.zeros((nslice,npi,npi*2))

        for frame in xrange(nframe):
                sys.stdout.write("CREATING ANGLE POPULATION GRID from {} out of {} frames\r".format(frame, nframe) )
                sys.stdout.flush()

                if ow_angles:
                        angles = mol_angles(directory, frame, nslice, nmol, model, csize, COM, DIM, nsite)
                        z_array, theta, phi, varphi = angles
                else:

			if os.path.exists('{}/EULER/{}_{}_ANGLE.txt'.format(directory, model.lower(), frame)): 
				ut.convert_txt_npy('{}/EULER/{}_{}_ANGLE'.format(directory, model.lower(), frame))
                        try:
                                with file('{}/EULER/{}_{}_ANGLE.npy'.format(directory, model.lower(), frame), 'r') as infile:
                                        z_array, theta, phi, varphi = np.load(infile)

                        except Exception:
                                angles = mol_angles(directory, frame, nslice, nmol, model, csize, COM, DIM, nsite)
                                z_array, theta, phi, varphi = angles

                if model.upper() not in ['ETHANOL', 'METHANOL', 'DMSO']:
                        phi = abs(phi)

                for j in xrange(nmol):

                        z = z_array[j] + DIM[2]/2.
                        index1 = int(z * nslice / DIM[2]) % nslice
                        index2 = int(theta[j] / dpi)
                        index3 = int((phi[j] + np.pi / 2.) / dpi)

			try: P_z_theta_phi[index1][index2][index3] += 1
                        except IndexError: pass

        print ""
        print "NORMALISING GRID"
        for index1 in xrange(nslice):
                if np.sum(P_z_theta_phi[index1]) != 0:
                        P_z_theta_phi[index1] = P_z_theta_phi[index1] / np.sum(P_z_theta_phi[index1])

        P_z_phi_theta = np.rollaxis(np.rollaxis(P_z_theta_phi, 2), 1)

        X_theta = np.arange(0, np.pi, dpi)
        X_phi = np.arange(-np.pi / 2, np.pi / 2, dpi)

        av_theta = np.zeros(nslice)
        av_phi = np.zeros(nslice)
        P1 = np.zeros(nslice)
        P2 = np.zeros(nslice)

        print "BUILDING AVERAGE ANGLE PROFILES"

        for index1 in xrange(nslice):
                sys.stdout.write("PROCESSING AVERAGE ANGLE PROFILES {} out of {} slices\r".format(index1, nslice) )
                sys.stdout.flush()

                for index2 in xrange(npi):
                        av_theta[index1] += np.sum(P_z_theta_phi[index1][index2]) * X_theta[index2]
                        P1[index1] += np.sum(P_z_phi_theta[index1][index2]) * np.cos(X_theta[index2])
                        P2[index1] += np.sum(P_z_phi_theta[index1][index2]) * 0.5 * (3 * np.cos(X_theta[index2])**2 - 1)

                        av_phi[index1] += np.sum(P_z_phi_theta[index1][index2]) * (X_phi[index2])

		if av_theta[index1] == 0:
                        av_theta[index1] += np.pi / 2.
                        av_phi[index1] += np.pi / 4.

        a_dist = (av_theta, av_phi, P1, P2)

        return a_dist


def polarisability(directory, model, csize, nframe, nmol, nsite, COM, DIM, nslice, npi, a, ow_polar):

        count_O = np.zeros(nslice)

        av_O = np.zeros((nslice, 9))

	vd = DIM[0] * DIM[1] * DIM[2] / nslice

        for frame in xrange(nframe):
                sys.stdout.write("CREATING ROTATIONAL TENSOR PROFILE from {} out of {} frames\r".format(frame, nframe) )
                sys.stdout.flush()

                with file('{}/DEN/{}_{}_{}_COUNT.npy'.format(directory, model.lower(), nslice, frame)) as infile:
                        count_array = np.load(infile)

                count_O += count_array[-1]

                if ow_polar: temp_O = R_tensors(directory, frame, nslice, nmol, model, csize, COM, DIM, nsite)
                else:
			if os.path.exists('{}/EULER/{}_{}_{}_ODIST.txt'.format(directory, model.lower(), nslice, frame)):
                                ut.convert_txt_npy('{}/EULER/{}_{}_{}_ODIST'.format(directory, model.lower(), nslice, frame))
                        try:
                                with file('{}/EULER/{}_{}_{}_ODIST.npy'.format(directory, model.lower(), nslice, frame), 'r') as infile:
                                        temp_O = np.load(infile)

                        except Exception: temp_O = R_tensors(directory, frame, nslice, nmol, model, csize, COM, DIM, nsite)

                av_O += temp_O

        axx = np.zeros(nslice)
        azz = np.zeros(nslice)

        for n in xrange(nslice):
                if count_O[n] != 0:
                        av_O[n] *= 1./ count_O[n]
                        for j in xrange(3):
                                axx[n] += a[j] * 0.5 * (av_O[n][j] + av_O[n][j+3])
                                azz[n] += a[j] * av_O[n][j+6]
                else:
                        axx[n] = np.mean(a)
                        azz[n] = np.mean(a)

        polar = (axx, azz)

        return polar


def euler_profile(directory, nframe, nslice, nmol, model, csize, suffix, AT, Q, M, LJ, COM, DIM, nsite, a_type, npi, ow_angle, ow_polar):
	print ""
	print "CALCULATING ORIENTATIONS OF {} {} SIZE {}\n".format(model.upper(), suffix.upper(), csize)	

	dpi = np.pi / npi
	av_varphi = np.zeros(nslice)

	a = ut.get_polar_constants(model, a_type)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice)

	with file('{}/DEN/{}_{}_{}_DEN.npy'.format(directory, model.lower(), nslice, nframe), 'r') as infile:
		av_density = np.load(infile)

	axx, azz = polarisability(directory, model, csize, nframe, nmol, nsite, COM, DIM, nslice, npi, a, ow_polar)
	av_theta, av_phi, P1, P2  = angle_dist(directory, model, csize, nframe, nmol, nsite, COM, DIM, nslice, npi, ow_angle)

	with file('{}/EULER/{}_{}_{}_{}_EUL.npy'.format(directory, model.lower(), a_type, nslice, nframe), 'w') as outfile:
		np.save(outfile, (axx, azz, av_theta, av_phi, av_varphi, P1, P2))

#test_orientation()
