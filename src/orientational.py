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

def local_frame(directory, xat, yat, zat, nmol, frame, model, nsite, eig_vec):

	sys.stdout.write("PROCESSING {} ODIST {} \r".format(directory, frame))
        sys.stdout.flush()

	xat_mol = xat.reshape((nmol, nsite))
	yat_mol = yat.reshape((nmol, nsite))
	zat_mol = zat.reshape((nmol, nsite))

	molecules = np.stack((xat_mol, yat_mol, zat_mol), axis=2)

	O = ut.local_frame_molecule(molecules, model, eig_vec)
	
	temp_O = O.reshape((nmol, 9))

	"""
        for j in xrange(nmol):
                sys.stdout.write("PROCESSING {} ODIST {}: {} out of {}  molecules\r".format(directory, frame, j, nmol))
                sys.stdout.flush()

		O = ut.local_frame_molecule(molecules[j], model, eig_vec)
                if O[2][2] < -1: O[2][2] = -1.0
                elif O[2][2] > 1: O[2][2] = 1.0
		temp_O[j] = O.reshape(9)

	print O_mol[-1] - O

	print temp_O_mol[-1] - temp_O[-1]
	"""
	return temp_O


def R_tensors(directory, O_frame, zmol, frame, nslice, nmol, DIM):

	sys.stdout.write("PROCESSING {} RDIST {}\r".format(directory, frame))
        sys.stdout.flush()

	temp_R = np.zeros((nslice, 9))

	index_z = np.array((zmol + DIM[2]/2) * nslice / DIM[2], dtype=int) % nslice

        for j in xrange(nmol):
            
		temp_R[index_z[j]] += O_frame[j]**2

        return temp_R


def mol_angles(directory, O_frame, zmol, frame, nmol):

	O = np.moveaxis(O_frame, 0, 1)

	theta = np.arccos(O[8])
        phi = (np.arctan(-O[6] / O[7]))
        varphi =  (np.arctan(O[2] / O[5]))

	"""
        theta = np.zeros(nmol)
        phi = np.zeros(nmol)
        varphi = np.zeros(nmol)

	for j in xrange(nmol):
                sys.stdout.write("PROCESSING {} ANGLES {}:  {} out of {}  molecules\r".format(directory, frame, j, nmol) )
                sys.stdout.flush()

                O = O_frame[j]

                theta[j] = np.arccos(O[8])
                phi[j] = (np.arctan(-O[6] / O[7]))
                varphi[j] =  (np.arctan(O[2] / O[5]))

	print theta[-1], theta_mol[-1]
	"""
        return theta, phi, varphi


def angle_dist(directory, model, csize, nframe, ntraj, nmol, nsite, com, DIM, nslice, npi, eig_vec, ow_angle):

        print "BUILDING ANGLE DISTRIBUTIONS"

        dpi = np.pi / npi

	file_name_odist = '{}_{}'.format(model.lower(), ntraj)
	file_name_angle = '{}_{}'.format(model.lower(), nframe)
	file_name_pangle = '{}_{}_{}_{}'.format(model.lower(), nslice, npi, nframe)

	if not ow_angle:
		try:
			with file('{}/EULER/{}_PANGLE.npy'.format(directory, file_name_pangle), 'r') as infile:
				P_z_theta_phi = np.load(infile)
		except IOError: ow_angle = True

	if ow_angle:
	
		with file('{}/EULER/{}_ODIST.npy'.format(directory, file_name_odist), 'r') as infile:
			tot_O = np.load(infile)

		_, _, zmol = ut.read_mol_positions(directory, model, csize, ntraj, nframe, com)
		COM = ut.read_com_positions(directory, model, csize, ntraj, nframe, com)
        	tot_theta = np.zeros((nframe, nmol))
        	tot_phi = np.zeros((nframe, nmol))
        	tot_varphi = np.zeros((nframe, nmol))
        	P_z_theta_phi = np.zeros((nslice, npi, npi*2))

		for frame in xrange(nframe):
		        sys.stdout.write("CREATING ANGLE POPULATION GRID from {} out of {} frames\r".format(frame, nframe) )
		        sys.stdout.flush()

			if os.path.exists('{}/EULER/{}_{}_ANGLE.txt'.format(directory, model.lower(), frame)): 
				ut.convert_txt_npy('{}/EULER/{}_{}_ANGLE'.format(directory, model.lower(), frame))
	                try:
	                        with file('{}/EULER/{}_{}_ANGLE.npy'.format(directory, model.lower(), frame), 'r') as infile:
	                                z_array, theta, phi, varphi = np.load(infile)
				os.remove('{}/EULER/{}_{}_ANGLE.npy'.format(directory, model.lower(), frame))

	                except Exception:
	                        angles = mol_angles(directory, tot_O[frame], zmol[frame]-COM[frame][2], frame, nmol)
	                        theta, phi, varphi = angles

			tot_theta[frame] += theta
			tot_phi[frame] += phi
			tot_varphi[frame] += varphi

		        if model.upper() not in ['ETHANOL', 'METHANOL', 'DMSO']:
		                phi = abs(phi)

	                z = zmol[frame] - COM[frame][2] + DIM[2]/2.
	                index1 = np.array(z * nslice / DIM[2], dtype=int) % nslice
	                index2 = np.array(theta / dpi, dtype=int)
	                index3 = np.array((phi + np.pi / 2.) / dpi, dtype=int)

			P_z_theta_phi += np.histogramdd((index1, index2, index3), bins=(nslice, npi, npi*2), range=[[0, nslice], [0, npi], [0, npi*2]])[0]

		with file('{}/EULER/{}_THETA.npy'.format(directory, file_name_angle), 'w') as outfile:
			np.save(outfile, tot_theta)
		with file('{}/EULER/{}_PHI.npy'.format(directory, file_name_angle), 'w') as outfile:
			np.save(outfile, tot_phi)
		with file('{}/EULER/{}_VARPHI.npy'.format(directory, file_name_angle), 'w') as outfile:
			np.save(outfile, tot_varphi)
		with file('{}/EULER/{}_PANGLE.npy'.format(directory, file_name_pangle), 'w') as outfile:
			np.save(outfile, P_z_theta_phi)

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


def polarisability(directory, model, csize, nframe, ntraj, nmol, nsite, com, DIM, nslice, npi, eig_val, eig_vec, ow_polar):

	vd = DIM[0] * DIM[1] * DIM[2] / nslice

	file_name_odist = '{}_{}'.format(model.lower(), ntraj)
	file_name_rdist = '{}_{}_{}'.format(model.lower(), nslice, nframe)

	if os.path.exists('{}/EULER/{}_ODIST.npy'.format(directory, file_name_rdist)): os.remove('{}/EULER/{}_ODIST.npy'.format(directory, file_name_rdist))

	if not ow_polar:
		try:
			with file('{}/EULER/{}_RDIST.npy'.format(directory, file_name_rdist), 'r') as infile:
				tot_R = np.load(infile)
		except IOError: ow_polar = True

	if ow_polar:

		with file('{}/EULER/{}_ODIST.npy'.format(directory, file_name_odist), 'r') as infile:
			tot_O = np.load(infile)

		tot_R = np.zeros((nframe, nslice, 9))
		_, _, zmol = ut.read_mol_positions(directory, model, csize, ntraj, nframe, com)
		COM = ut.read_com_positions(directory, model, csize, ntraj, nframe, com)

		for frame in xrange(nframe):
		        sys.stdout.write("CREATING ROTATIONAL TENSOR PROFILE from {} out of {} frames\r".format(frame, nframe) )
		        sys.stdout.flush()

			if os.path.exists('{}/EULER/{}_{}_{}_ODIST.txt'.format(directory, model.lower(), nslice, frame)):
		        	ut.convert_txt_npy('{}/EULER/{}_{}_{}_ODIST'.format(directory, model.lower(), nslice, frame))

			try:
	                        with file('{}/EULER/{}_{}_{}_ODIST.npy'.format(directory, model.lower(), nslice, frame), 'r') as infile:
	                                temp_R = np.load(infile)
				os.remove('{}/EULER/{}_{}_{}_ODIST.npy'.format(directory, model.lower(), nslice, frame))
	                except:
				temp_R = R_tensors(directory, tot_O[frame], zmol[frame] - COM[frame][2], frame, nslice, nmol, DIM)

			tot_R[frame] += temp_R

		with file('{}/EULER/{}_RDIST.npy'.format(directory, file_name_rdist), 'w') as outfile:
		        np.save(outfile, tot_R)

	with file('{}/DEN/{}_COUNT.npy'.format(directory, file_name_rdist), 'r') as infile:
		tot_count_array = np.load(infile)

	av_R = np.sum(tot_R, axis=0)
	count_O = np.sum(tot_count_array, axis=0)[-1]

        axx = np.zeros(nslice)
        azz = np.zeros(nslice)

        for n in xrange(nslice):
                if count_O[n] != 0:
                        av_R[n] *= 1./ count_O[n]
                        for j in xrange(3):
                                axx[n] += eig_val[j] * 0.5 * (av_R[n][j] + av_R[n][j+3])
                                azz[n] += eig_val[j] * av_R[n][j+6]
                else:
                        axx[n] = np.mean(eig_val)
                        azz[n] = np.mean(eig_val)

	plt.scatter(np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice), axx, c='b')
	plt.scatter(np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice), azz, c='r')
	plt.show()

        polar = (axx, azz)

        return polar


def euler_profile(directory, nframe, ntraj, nslice, nmol, model, csize, suffix, AT, Q, M, LJ, com, DIM, nsite, a_type, npi, ow_local, ow_angle, ow_polar):
	print ""
	print "CALCULATING ORIENTATIONS OF {} {} SIZE {}\n".format(model.upper(), suffix.upper(), csize)	

	dpi = np.pi / npi
	av_varphi = np.zeros(nslice)

	eig_val, eig_vec = ut.get_polar_constants(model, a_type)

	Z1 = np.linspace(0, DIM[2], nslice)
	Z2 = np.linspace(-1/2.*DIM[2], 1/2.*DIM[2], nslice)
	
	file_name_odist = '{}_{}'.format(model.lower(), ntraj)

	if not os.path.exists('{}/EULER/{}_ODIST.npy'.format(directory, file_name_odist)) or ow_local:
		xat, yat, zat = ut.read_atom_positions(directory, model, csize, ntraj, com)
		tot_O = np.zeros((ntraj, nmol, 9))
		for frame in xrange(ntraj):
			tot_O[frame] += local_frame(directory, xat[frame], yat[frame], zat[frame], nmol, frame, model, nsite, eig_vec)
		with file('{}/EULER/{}_ODIST.npy'.format(directory, file_name_odist), 'w') as outfile:
			np.save(outfile, tot_O)
	
	axx, azz = polarisability(directory, model, csize, nframe, ntraj, nmol, nsite, com, DIM, nslice, npi, eig_val, eig_vec, ow_polar)
	av_theta, av_phi, P1, P2  = angle_dist(directory, model, csize, nframe, ntraj, nmol, nsite, com, DIM, nslice, npi, eig_vec, ow_angle)

	with file('{}/EULER/{}_{}_{}_{}_EUL.npy'.format(directory, model.lower(), a_type, nslice, nframe), 'w') as outfile:
		np.save(outfile, (axx, azz, av_theta, av_phi, av_varphi, P1, P2))

#test_orientation()
